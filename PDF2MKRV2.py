import asyncio
import os
import sys
import aiohttp
import aiofiles
from playwright.async_api import async_playwright, TimeoutError
from PIL import Image
from pypdf import PdfWriter
from io import BytesIO

# --- Configuration ---
Image.MAX_IMAGE_PIXELS = None
PIXEL_LIMIT_PER_STRIP = 65000
PARALLEL_JOBS = 4
QUALITY_MULTIPLIER = 2
CACHE_DIR = "script_cache"
VIEWPORT_HEIGHT = 1200
# A very generous timeout for the screenshot command itself, for very slow pages.
SCREENSHOT_TIMEOUT = 180000  # 3 minutes

# --- Dependencies ---
DEPENDENCIES = {
    "https://cdn.plot.ly/plotly-2.32.0.min.js": "plotly.js",
    "https://d3js.org/d3.v7.min.js": "d3.js",
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js": "mathjax.js",
}


async def ensure_dependencies_cached():
    """Checks if JavaScript dependencies are cached and downloads them if not."""
    print("--- Checking Dependencies ---")
    os.makedirs(CACHE_DIR, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for url, filename in DEPENDENCIES.items():
            local_path = os.path.join(CACHE_DIR, filename)
            if not os.path.exists(local_path):
                print(f"'{filename}' not found. Downloading...")
                try:
                    async with session.get(url, timeout=60) as response:
                        response.raise_for_status()
                        content = await response.read()
                        async with aiofiles.open(local_path, 'wb') as f:
                            await f.write(content)
                        print(f"  ✅ Cached '{filename}'")
                except Exception as e:
                    print(f"  ❌ Failed to download '{filename}': {e}")
                    return False
    print("--- Dependency check complete. ---")
    return True


async def wait_for_page_to_load(page, base_filename):
    """Contains the robust waiting logic for all dynamic content to finish loading."""
    total_height = await page.evaluate("document.body.scrollHeight")
    for i in range(0, total_height, VIEWPORT_HEIGHT):
        await page.evaluate(f"window.scrollTo(0, {i})")
        await page.wait_for_timeout(100)
    await page.evaluate("window.scrollTo(0, 0)")

    try:
        await page.wait_for_load_state('networkidle', timeout=30000)
    except TimeoutError:
        print(f"  - Note: Network idle timed out for {base_filename}.")
    try:
        await page.wait_for_function("typeof MathJax?.startup?.promise.then(p => p && p.done) !== 'undefined'", timeout=60000)
    except TimeoutError:
        print(f"  - Warning: MathJax timed out for {base_filename}.")
    try:
        await page.wait_for_function(
            """() => {
                const plots = Array.from(document.querySelectorAll('.js-plotly-plot'));
                return !plots.length || plots.every(p => p.querySelector('.main-svg') && p.querySelector('.main-svg').clientHeight > 10);
            }""",
            timeout=60000
        )
    except TimeoutError:
        if await page.query_selector('.js-plotly-plot'):
            print(f"  - Warning: Plotly timed out for {base_filename}.")
    await page.wait_for_timeout(1000)


async def process_single_file(index, total, semaphore, browser, html_file_path, output_directory, cached_urls):
    """
    Processes one HTML file, with an extremely patient timeout for screenshots
    and a robust fallback for very long pages.
    """
    async with semaphore:
        base_filename = os.path.splitext(os.path.basename(html_file_path))[0]
        final_pdf_path = os.path.join(output_directory, f"{base_filename}.pdf")

        if os.path.exists(final_pdf_path):
            print(f"⏭️  [{index}/{total}] Skipping (already exists): {os.path.basename(final_pdf_path)}")
            return

        print(f"▶️  [{index}/{total}] Processing: {os.path.basename(html_file_path)}")
        context, page = None, None
        try:
            context = await browser.new_context(
                viewport={"width": 1280, "height": VIEWPORT_HEIGHT},
                device_scale_factor=QUALITY_MULTIPLIER
            )
            page = await context.new_page()
            
            async def handle_route(route):
                if route.request.url in cached_urls:
                    await route.fulfill(path=os.path.join(CACHE_DIR, DEPENDENCIES[route.request.url]))
                else:
                    await route.continue_()

            await page.route("**/*", handle_route)
            await page.goto(f'file://{html_file_path}', wait_until='domcontentloaded', timeout=90000)
            await wait_for_page_to_load(page, base_filename)

            try:
                # STRATEGY 1: Attempt fast, full-page screenshot with long timeout
                png_data = await page.screenshot(full_page=True, timeout=SCREENSHOT_TIMEOUT)
                with Image.open(BytesIO(png_data)) as img:
                    img.save(final_pdf_path, "PDF", resolution=100.0)
                print(f"  ✅ Created full-page PDF: {os.path.basename(final_pdf_path)}")

            except Exception as e:
                # STRATEGY 2: Fallback for very long pages
                if "Maximum supported image dimension" in str(e) or "broken data stream" in str(e):
                    print(f"  - Note: Page is too long, falling back to stitching method for {base_filename}.")
                    
                    image_data_list = []
                    total_height = await page.evaluate("document.body.scrollHeight")
                    for i, offset in enumerate(range(0, total_height, VIEWPORT_HEIGHT)):
                        print(f"    -> Capturing chunk {i+1}...")
                        await page.evaluate(f"window.scrollTo(0, {offset})")
                        await page.wait_for_timeout(50)
                        # Use the same long timeout for each chunk
                        image_data_list.append(await page.screenshot(timeout=SCREENSHOT_TIMEOUT))

                    images = [Image.open(BytesIO(data)) for data in image_data_list]
                    
                    pdf_merger = PdfWriter()
                    current_strip_height = 0
                    current_strip_images = []

                    for img in images:
                        if current_strip_height + img.height > PIXEL_LIMIT_PER_STRIP:
                            strip = Image.new('RGB', (img.width, current_strip_height))
                            y = 0
                            for part in current_strip_images: strip.paste(part, (0, y)); y += part.height
                            with BytesIO() as pdf_buffer:
                                strip.save(pdf_buffer, "PDF", resolution=100.0)
                                pdf_merger.append(BytesIO(pdf_buffer.getvalue()))
                            current_strip_images, current_strip_height = [img], img.height
                        else:
                            current_strip_images.append(img); current_strip_height += img.height

                    if current_strip_images:
                        strip = Image.new('RGB', (images[0].width, current_strip_height))
                        y = 0
                        for part in current_strip_images: strip.paste(part, (0, y)); y += part.height
                        with BytesIO() as pdf_buffer:
                            strip.save(pdf_buffer, "PDF", resolution=100.0)
                            pdf_merger.append(BytesIO(pdf_buffer.getvalue()))
                    
                    with open(final_pdf_path, 'wb') as f: pdf_merger.write(f)
                    pdf_merger.close()
                    print(f"  ✅ Created stitched PDF: {os.path.basename(final_pdf_path)}")
                else:
                    raise e

        except Exception as e:
            print(f"  ❌ An unexpected error occurred while processing {base_filename}: {e}")
        finally:
            if page: await page.close()
            if context: await context.close()


async def main():
    if len(sys.argv) < 2:
        print(f"❌ Error: No folder path provided.\n   Usage: python {os.path.basename(__file__)} \"C:\\path\\to\\html_files\"")
        return
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print(f"❌ Invalid path: '{directory_path}' is not a valid folder.")
        return

    if not await ensure_dependencies_cached():
        print("Could not retrieve all dependencies. Aborting.")
        return

    html_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.html')]
    if not html_files:
        print(f"❌ No HTML files found in '{directory_path}'")
        return

    total = len(html_files)
    semaphore = asyncio.Semaphore(PARALLEL_JOBS)
    cached_urls = set(DEPENDENCIES.keys())
    print(f"\n🔄 Found {total} files. Converting with {PARALLEL_JOBS} parallel jobs...\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
        tasks = [process_single_file(i + 1, total, semaphore, browser, file_path, directory_path, cached_urls) for i, file_path in enumerate(html_files)]
        await asyncio.gather(*tasks)
        await browser.close()

    print("\n✅ All files converted successfully!")


if __name__ == '__main__':
    asyncio.run(main())
