import os
import sys
import fitz  # PyMuPDF
import json
from tqdm import tqdm
import uuid

# --- Configuration ---
TARGET_JSON_VERSION = "0.2.0-beta.6"
TARGET_PAGE_VERSION = "0.1.6"
DEFAULT_ZOOM = 1.0
# --- END OF CONFIGURATION ---

class PostMergedCandidate:
    # This class is correct and does not need changes from the last version.
    def __init__(self, initial_json_block: dict):
        self.blocks_in_group = [initial_json_block]
        self.min_x0, self.min_y0, self.max_x1, self.max_y1 = map(float, initial_json_block['box'])
        self.font_sizes_in_group = [float(initial_json_block['font_size'])]
        self.is_vertical = initial_json_block['vertical']
        self.constituent_lines_list = [list(initial_json_block['lines'])]

    def add_json_block(self, json_block_to_add: dict):
        self.blocks_in_group.append(json_block_to_add)
        b_x0, b_y0, b_x1, b_y1 = map(float, json_block_to_add['box'])
        self.min_x0 = min(self.min_x0, b_x0)
        self.min_y0 = min(self.min_y0, b_y0)
        self.max_x1 = max(self.max_x1, b_x1)
        self.max_y1 = max(self.max_y1, b_y1)
        self.font_sizes_in_group.append(float(json_block_to_add['font_size']))
        self.constituent_lines_list.append(list(json_block_to_add['lines']))

    @property
    def bbox(self): return [self.min_x0, self.min_y0, self.max_x1, self.max_y1]
    @property
    def avg_font_size(self):
        if not self.font_sizes_in_group: return 10.0
        return round(sum(self.font_sizes_in_group) / len(self.font_sizes_in_group), 1)
    @property
    def avg_char_width_estimate(self): return self.avg_font_size * 0.8

    def finalize(self):
        final_box = [int(coord) for coord in self.bbox]
        final_lines_list = []
        for sublist in self.constituent_lines_list:
            final_lines_list.extend(sublist)
        cleaned_lines = [line.strip() for line in final_lines_list]
        final_lines_list = [line for line in cleaned_lines if line]
        return {
            "box": final_box, "vertical": self.is_vertical,
            "font_size": self.avg_font_size, "lines": final_lines_list
        }


def extract_blocks_from_page_directly(page, debug_this_page=False):
    """
    NEW AND IMPROVED: Hybrid approach.
    Uses PyMuPDF's block grouping but analyzes orientation on a line-by-line basis
    to correctly handle mixed-orientation blocks.
    """
    page_dict = page.get_text("dict", sort=False) # Sorting is handled later
    initial_blocks = []

    for pymupdf_block in page_dict["blocks"]:
        if pymupdf_block.get("type", 1) != 0: continue # Not a text block

        # Store lines classified by orientation
        vertical_lines_data = []
        horizontal_lines_data = []

        for line_obj in pymupdf_block.get("lines", []):
            line_text = "".join(s["text"] for s in line_obj["spans"]).strip()
            if not line_text:
                continue

            # Determine line orientation
            # 1. Primary check: wmode property
            is_line_vertical = line_obj.get("wmode") == 1
            # 2. Secondary check: flags in spans
            if not is_line_vertical:
                if any(span.get("flags", 0) & 4 for span in line_obj["spans"]):
                    is_line_vertical = True
            
            # Collect line data
            line_font_sizes = [s["size"] for s in line_obj["spans"]]
            avg_line_fs = round(sum(line_font_sizes) / len(line_font_sizes), 1) if line_font_sizes else 10.0
            
            line_data = {
                "text": line_text,
                "bbox": line_obj["bbox"],
                "size": avg_line_fs,
            }

            if is_line_vertical:
                vertical_lines_data.append(line_data)
            else:
                horizontal_lines_data.append(line_data)

        # Process the collected lines for this block into separate sub-blocks if needed
        if vertical_lines_data:
            # Combine all vertical lines into a single block
            min_x0 = min(d["bbox"][0] for d in vertical_lines_data)
            min_y0 = min(d["bbox"][1] for d in vertical_lines_data)
            max_x1 = max(d["bbox"][2] for d in vertical_lines_data)
            max_y1 = max(d["bbox"][3] for d in vertical_lines_data)
            avg_fs = round(sum(d["size"] for d in vertical_lines_data) / len(vertical_lines_data), 1)
            
            # Sort vertical lines top-to-bottom for correct reading order
            vertical_lines_data.sort(key=lambda d: d["bbox"][1])

            initial_blocks.append({
                "box": [int(min_x0), int(min_y0), int(max_x1), int(max_y1)],
                "vertical": True,
                "font_size": avg_fs,
                "lines": [d["text"] for d in vertical_lines_data]
            })

        if horizontal_lines_data:
            # Combine all horizontal lines into a single block
            min_x0 = min(d["bbox"][0] for d in horizontal_lines_data)
            min_y0 = min(d["bbox"][1] for d in horizontal_lines_data)
            max_x1 = max(d["bbox"][2] for d in horizontal_lines_data)
            max_y1 = max(d["bbox"][3] for d in horizontal_lines_data)
            avg_fs = round(sum(d["size"] for d in horizontal_lines_data) / len(horizontal_lines_data), 1)
            
            # Sort horizontal lines top-to-bottom
            horizontal_lines_data.sort(key=lambda d: d["bbox"][1])
            
            initial_blocks.append({
                "box": [int(min_x0), int(min_y0), int(max_x1), int(max_y1)],
                "vertical": False,
                "font_size": avg_fs,
                "lines": [d["text"] for d in horizontal_lines_data]
            })

    # Sort all generated blocks by reading order for the final merge pass
    initial_blocks.sort(key=lambda b: (
        -b["box"][0] if b["vertical"] else b["box"][1],  # RTL for vertical, TTB for horizontal
        b["box"][1] if b["vertical"] else b["box"][0]    # TTB for vertical, LTR for horizontal
    ))

    if debug_this_page:
        print(f"  Hybrid Extraction: Created {len(initial_blocks)} orientation-pure blocks before merging.")
    return initial_blocks


def can_merge_json_candidates(current_candidate: PostMergedCandidate,
                              next_json_block: dict,
                              debug_this_page=False):
    if current_candidate.is_vertical != next_json_block['vertical']:
        if debug_this_page: print(f"      PostMerge: Orientation mismatch. No merge.")
        return False

    g_x0, g_y0, g_x1, g_y1 = current_candidate.bbox
    n_x0, n_y0, n_x1, n_y1 = map(float, next_json_block['box'])

    ref_dim_group = current_candidate.avg_char_width_estimate
    ref_dim_next = float(next_json_block['font_size']) * 0.8
    avg_ref_dim = (ref_dim_group + ref_dim_next) / 2.0 if ref_dim_group > 0 and ref_dim_next > 0 else 10.0

    PRIMARY_AXIS_GAP_MULTIPLIER = 1.8
    SECONDARY_AXIS_ALIGNMENT_MULTIPLIER = 1.3
    MIN_SECONDARY_AXIS_OVERLAP_RATIO = 0.05

    if current_candidate.is_vertical:
        horizontal_gap = g_x0 - n_x1
        allowed_h_gap = PRIMARY_AXIS_GAP_MULTIPLIER * avg_ref_dim
        y_center_diff = abs(((g_y0 + g_y1) / 2) - ((n_y0 + n_y1) / 2))
        allowed_y_center_diff = SECONDARY_AXIS_ALIGNMENT_MULTIPLIER * avg_ref_dim
        vertical_overlap_height = max(0, min(g_y1, n_y1) - max(g_y0, n_y0))
        min_col_height = min(g_y1 - g_y0, n_y1 - n_y0)
        is_horizontally_proximate = (-allowed_h_gap * 0.25 < horizontal_gap < allowed_h_gap)
        is_vertically_aligned = (vertical_overlap_height > MIN_SECONDARY_AXIS_OVERLAP_RATIO * min_col_height if min_col_height > 0 else False) or (y_center_diff < allowed_y_center_diff)
        return is_horizontally_proximate and is_vertically_aligned
    else:
        vertical_gap = n_y0 - g_y1
        allowed_v_gap = PRIMARY_AXIS_GAP_MULTIPLIER * avg_ref_dim
        x_center_diff = abs(((g_x0 + g_x1) / 2) - ((n_x0 + n_x1) / 2))
        allowed_x_center_diff = SECONDARY_AXIS_ALIGNMENT_MULTIPLIER * avg_ref_dim
        horizontal_overlap_width = max(0, min(g_x1, n_x1) - max(g_x0, n_x0))
        min_line_width = min(g_x1 - g_x0, n_x1 - n_x0)
        is_vertically_proximate = (-allowed_v_gap * 0.5 < vertical_gap < allowed_v_gap)
        is_horizontally_aligned = (horizontal_overlap_width > MIN_SECONDARY_AXIS_OVERLAP_RATIO * min_line_width if min_line_width > 0 else False) or (x_center_diff < allowed_x_center_diff)
        return is_vertically_proximate and is_horizontally_aligned

def merge_json_blocks_post_process(initial_json_blocks: list, debug_this_page=False):
    if not initial_json_blocks: return []
    if len(initial_json_blocks) == 1:
        # If only one block, finalize it and return
        candidate = PostMergedCandidate(initial_json_blocks[0])
        finalized_block = candidate.finalize()
        return [finalized_block] if finalized_block.get('lines') else []

    if debug_this_page: print(f"  Starting post-process merging for {len(initial_json_blocks)} initial blocks.")
    merged_final_blocks = []
    current_candidate_group = PostMergedCandidate(initial_json_blocks[0])

    for i in range(1, len(initial_json_blocks)):
        next_block_to_consider = initial_json_blocks[i]
        if can_merge_json_candidates(current_candidate_group, next_block_to_consider, debug_this_page):
            current_candidate_group.add_json_block(next_block_to_consider)
        else:
            finalized_group_block = current_candidate_group.finalize()
            if finalized_group_block['lines']:
                merged_final_blocks.append(finalized_group_block)
            current_candidate_group = PostMergedCandidate(next_block_to_consider)

    if current_candidate_group:
        finalized_group_block = current_candidate_group.finalize()
        if finalized_group_block['lines']:
            merged_final_blocks.append(finalized_group_block)

    if debug_this_page: print(f"  Post-process merging complete. Final block count: {len(merged_final_blocks)}")
    return merged_final_blocks

def structure_page_for_json_output(page, zoom_factor=1.0, debug_this_page=False):
    mat = fitz.Matrix(zoom_factor, zoom_factor)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    initial_blocks = extract_blocks_from_page_directly(page, debug_this_page=debug_this_page)
    final_merged_blocks = merge_json_blocks_post_process(initial_blocks, debug_this_page=debug_this_page)
    
    page_data_for_json = {
        "version": TARGET_PAGE_VERSION,
        "img_width": int(page.rect.width),
        "img_height": int(page.rect.height),
        "blocks": final_merged_blocks,
    }
    return page_data_for_json, pix


def process_pdf_to_target_json(pdf_path, output_dir, zoom_factor=DEFAULT_ZOOM, debug_page_index=-1):
    try:
        with fitz.open(pdf_path) as doc:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            pdf_title = base_name
            pdf_volume_name = base_name
            title_uuid_str = str(uuid.uuid4())
            volume_uuid_str = str(uuid.uuid4())
            abs_image_dir = os.path.join(output_dir, pdf_volume_name)
            os.makedirs(abs_image_dir, exist_ok=True)

            output_mokuro_data = {
                "version": TARGET_JSON_VERSION, "title": pdf_title,
                "title_uuid": title_uuid_str, "volume": pdf_volume_name,
                "volume_uuid": volume_uuid_str, "pages": []
            }
            print(f"Processing {base_name} ({len(doc)} pages) for target Mokuro format...")

            for i in tqdm(range(len(doc)), desc=f"Pages for {base_name}", unit="page"):
                page_obj = doc[i]
                debug_this_specific_page = (page_obj.number == debug_page_index)
                if debug_this_specific_page:
                    print(f"\n--- DEBUGGING PAGE {page_obj.number} (0-indexed: {i}) ---")

                page_json_data, pix_for_saving = structure_page_for_json_output(
                    page_obj, zoom_factor=zoom_factor,
                    debug_this_page=debug_this_specific_page
                )

                image_name = f"{i:03}.jpg"
                page_json_data["img_path"] = image_name
                image_save_path = os.path.join(abs_image_dir, image_name)
                try:
                    pix_for_saving.save(image_save_path)
                except Exception as e:
                    print(f"Warning: Could not save image for page {i} of {base_name}: {e}")
                output_mokuro_data["pages"].append(page_json_data)

            mokuro_file_path = os.path.join(output_dir, f"{pdf_volume_name}.mokuro")
            with open(mokuro_file_path, "w", encoding="utf-8") as f:
                json.dump(output_mokuro_data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            print(f"✓ Done. Output for {base_name} in: {output_dir}")
            print(f"  Mokuro file: {mokuro_file_path}")
            print(f"  Images: {abs_image_dir}")

    except fitz.FitzError as e:
        print(f"FitzError processing PDF {pdf_path}: {e}. Skipping this PDF.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {pdf_path}: {e}. Skipping this PDF.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not (2 <= len(sys.argv) <= 4):
        print(f"Usage: python {os.path.basename(__file__)} /path/to/pdf_or_directory [zoom_factor] [debug_page_index]")
        sys.exit(1)

    input_path_arg = sys.argv[1]
    zoom_factor_arg = DEFAULT_ZOOM
    debug_page_idx_arg = -1

    if len(sys.argv) >= 3:
        try:
            zoom_factor_arg = float(sys.argv[2])
            if zoom_factor_arg <= 0: raise ValueError("Zoom factor must be a positive number.")
        except ValueError as e:
            print(f"Error: Invalid zoom factor '{sys.argv[2]}'. {e}")
            sys.exit(1)
    
    if len(sys.argv) == 4:
        try:
            debug_page_idx_arg = int(sys.argv[3])
            if debug_page_idx_arg < 0:
                print("Warning: debug_page_index non-negative. Disabling debug.")
                debug_page_idx_arg = -1
        except ValueError:
            print(f"Error: Invalid debug_page_index '{sys.argv[3]}'. Disabling debug.")
            debug_page_idx_arg = -1

    pdf_files_to_process = []
    output_base_dir = ""

    if os.path.isfile(input_path_arg) and input_path_arg.lower().endswith(".pdf"):
        pdf_files_to_process.append(input_path_arg)
        output_base_dir = os.path.dirname(input_path_arg)
    elif os.path.isdir(input_path_arg):
        output_base_dir = input_path_arg
        for file_name_in_dir in os.listdir(input_path_arg):
            if file_name_in_dir.lower().endswith(".pdf"):
                pdf_files_to_process.append(os.path.join(input_path_arg, file_name_in_dir))
    else:
        print(f"Error: Input path '{input_path_arg}' is not a valid PDF file or directory.")
        sys.exit(1)

    if not pdf_files_to_process:
        print(f"No PDF files found at '{input_path_arg}'.")
        sys.exit(0)

    for pdf_file_path in pdf_files_to_process:
        process_pdf_to_target_json(
            pdf_file_path, output_base_dir,
            zoom_factor=zoom_factor_arg,
            debug_page_index=debug_page_idx_arg
        )