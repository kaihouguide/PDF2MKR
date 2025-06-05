import os
import sys
import fitz  # PyMuPDF
import json
from tqdm import tqdm
import uuid
import math # For ceiling division

# --- Configuration ---
TARGET_JSON_VERSION = "0.2.0-beta.6"
TARGET_PAGE_VERSION = "0.1.6"
DEFAULT_ZOOM = 1.0
# --- END OF CONFIGURATION ---

# --- Text Splitting Configuration ---
# These are a starting point, you'll likely need to tune them.
LINE_SPLIT_THRESHOLD_SHORT = 10  # If total text is longer than this, consider 2 lines
LINE_SPLIT_THRESHOLD_MEDIUM = 18 # If total text is longer than this, consider 3 lines
# Add more thresholds if you want more potential splits, e.g., LINE_SPLIT_THRESHOLD_LONG for 4 lines

def split_long_text(text: str, max_lines_hint: int = 0) -> list[str]:
    """
    Splits a long string into a few lines.
    Aims for roughly equal length lines.
    max_lines_hint can be used if a specific number of lines is desired (0 means auto).
    """
    text_len = len(text)
    if not text_len:
        return []

    num_lines = 1
    if max_lines_hint > 0:
        num_lines = max_lines_hint
    else: # Auto-determine based on length thresholds
        if text_len > LINE_SPLIT_THRESHOLD_MEDIUM:
            num_lines = 3
        elif text_len > LINE_SPLIT_THRESHOLD_SHORT:
            num_lines = 2
        # else num_lines remains 1

    if num_lines == 1 or text_len == 0:
        return [text]

    # Calculate ideal length per line, ensuring we get `num_lines`
    # Use math.ceil to ensure the last line isn't overly short if text_len is not perfectly divisible
    ideal_len_per_line = math.ceil(text_len / num_lines)
    
    result_lines = []
    current_pos = 0
    for i in range(num_lines):
        if current_pos >= text_len:
            break
        
        # For the last line, take all remaining text
        if i == num_lines - 1:
            segment = text[current_pos:]
        else:
            # Determine split point: ideal_len_per_line from current_pos
            # We could add more sophisticated logic here to find natural breaks (e.g., after particles)
            # For now, simple character count split.
            end_pos = min(current_pos + ideal_len_per_line, text_len)
            segment = text[current_pos:end_pos]
        
        if segment: # Ensure not adding empty strings if logic error
             result_lines.append(segment)
        current_pos += len(segment)
        
    # If due to ceiling or small text, we got fewer lines than intended,
    # but more than 1, that's acceptable. If only one, return it.
    if not result_lines: # Should not happen if text_len > 0
        return [text]
        
    return result_lines


class PostMergedCandidate:
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
        
        full_text_ordered_segments = []
        # The constituent_lines_list already has lists of lines,
        # and these lists are already in the merged order.
        for lines_sublist in self.constituent_lines_list:
            full_text_ordered_segments.extend(lines_sublist)
        
        concatenated_text = "".join(full_text_ordered_segments).strip()

        if not concatenated_text:
            final_lines_list = []
        else:
            # --- MODIFICATION: Use the new split_long_text function ---
            final_lines_list = split_long_text(concatenated_text)
            # --- END OF MODIFICATION ---

        return {
            "box": final_box, "vertical": self.is_vertical,
            "font_size": self.avg_font_size, "lines": final_lines_list
        }

# ... (rest of the script remains the same as the previous "fully fixed" version)
# Ensure can_merge_json_candidates, merge_json_blocks_post_process,
# extract_blocks_from_page_directly, structure_page_for_json_output,
# process_pdf_to_target_json, and if __name__ == "__main__":
# are all included below this point from the previous full script.
# I will paste it completely again to be sure.
# --- Merging Logic for Post-Processing ---
def can_merge_json_candidates(current_candidate: PostMergedCandidate,
                              next_json_block: dict,
                              debug_this_page=False):
    if current_candidate.is_vertical != next_json_block['vertical']:
        if debug_this_page: print(f"      PostMerge: Orientation mismatch (CandidateV: {current_candidate.is_vertical}, NextV: {next_json_block['vertical']}). No merge.")
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
        g_center_y = (g_y0 + g_y1) / 2
        n_center_y = (n_y0 + n_y1) / 2
        y_center_diff = abs(g_center_y - n_center_y)
        allowed_y_center_diff = SECONDARY_AXIS_ALIGNMENT_MULTIPLIER * avg_ref_dim
        overlap_y_start = max(g_y0, n_y0)
        overlap_y_end = min(g_y1, n_y1)
        vertical_overlap_height = overlap_y_end - overlap_y_start
        min_col_height = min(g_y1 - g_y0, n_y1 - n_y0) if (g_y1 - g_y0 > 0 and n_y1 - n_y0 > 0) else 0
        
        is_horizontally_proximate = (-allowed_h_gap * 0.25 < horizontal_gap < allowed_h_gap)
        is_vertically_aligned = (vertical_overlap_height > MIN_SECONDARY_AXIS_OVERLAP_RATIO * min_col_height if min_col_height > 0 else False) or \
                                (y_center_diff < allowed_y_center_diff)
        
        if debug_this_page:
            print(f"      PostMerge (V): Candidate box=[{g_x0:.0f},{g_y0:.0f},{g_x1:.0f},{g_y1:.0f}], Next box=[{n_x0:.0f},{n_y0:.0f},{n_x1:.0f},{n_y1:.0f}]")
            print(f"                       h_gap(RTL)={horizontal_gap:.1f} (allow ~{allowed_h_gap:.1f}), v_overlap={vertical_overlap_height:.1f}, y_center_diff={y_center_diff:.1f} (allow ~{allowed_y_center_diff:.1f})")
            print(f"                       is_h_prox={is_horizontally_proximate}, is_v_align={is_vertically_aligned}, avg_ref_dim={avg_ref_dim:.1f}")
        return is_horizontally_proximate and is_vertically_aligned
    else: 
        vertical_gap = n_y0 - g_y1 
        allowed_v_gap = PRIMARY_AXIS_GAP_MULTIPLIER * avg_ref_dim
        g_center_x = (g_x0 + g_x1) / 2
        n_center_x = (n_x0 + n_x1) / 2
        x_center_diff = abs(g_center_x - n_center_x)
        allowed_x_center_diff = SECONDARY_AXIS_ALIGNMENT_MULTIPLIER * avg_ref_dim
        overlap_x_start = max(g_x0, n_x0)
        overlap_x_end = min(g_x1, n_x1)
        horizontal_overlap_width = overlap_x_end - overlap_x_start
        min_line_width = min(g_x1 - g_x0, n_x1 - n_x0) if (g_x1 - g_x0 > 0 and n_x1 - n_x0 > 0) else 0

        is_vertically_proximate = (-allowed_v_gap * 0.5 < vertical_gap < allowed_v_gap)
        is_horizontally_aligned = (horizontal_overlap_width > MIN_SECONDARY_AXIS_OVERLAP_RATIO * min_line_width if min_line_width > 0 else False) or \
                                  (x_center_diff < allowed_x_center_diff)

        if debug_this_page:
            print(f"      PostMerge (H): Candidate box=[{g_x0:.0f},{g_y0:.0f},{g_x1:.0f},{g_y1:.0f}], Next box=[{n_x0:.0f},{n_y0:.0f},{n_x1:.0f},{n_y1:.0f}]")
            print(f"                       v_gap(TTB)={vertical_gap:.1f} (allow ~{allowed_v_gap:.1f}), h_overlap={horizontal_overlap_width:.1f}, x_center_diff={x_center_diff:.1f} (allow ~{allowed_x_center_diff:.1f})")
            print(f"                       is_v_prox={is_vertically_proximate}, is_h_align={is_horizontally_aligned}, avg_ref_dim={avg_ref_dim:.1f}")
        return is_vertically_proximate and is_horizontally_aligned
    return False

def merge_json_blocks_post_process(initial_json_blocks: list, debug_this_page=False):
    if not initial_json_blocks: return []
    if len(initial_json_blocks) < 2: return initial_json_blocks
    if debug_this_page: print(f"  Starting post-process merging for {len(initial_json_blocks)} initial blocks.")

    merged_final_blocks = []
    current_candidate_group = PostMergedCandidate(initial_json_blocks[0])
    if debug_this_page: 
        preview = "".join(b for sublist in current_candidate_group.constituent_lines_list for b in sublist)[:30].replace("\n"," ")
        print(f"    PostMerge: Initializing with first block. Candidate (text='{preview}...', box={current_candidate_group.bbox})")

    for i in range(1, len(initial_json_blocks)):
        next_block_to_consider = initial_json_blocks[i]
        if debug_this_page:
            cand_preview = "".join(b for sublist in current_candidate_group.constituent_lines_list for b in sublist)[:20].replace("\n"," ")
            next_preview = "".join(next_block_to_consider['lines'])[:20].replace("\n"," ")
            print(f"    PostMerge: Considering merge: Candidate (text='{cand_preview}...', box={current_candidate_group.bbox}) + Next (text='{next_preview}...', box={next_block_to_consider['box']})")

        if can_merge_json_candidates(current_candidate_group, next_block_to_consider, debug_this_page):
            if debug_this_page: print(f"      PostMerge: Merging Next into Candidate.")
            current_candidate_group.add_json_block(next_block_to_consider)
            if debug_this_page: 
                preview = "".join(b for sublist in current_candidate_group.constituent_lines_list for b in sublist)[:30].replace("\n"," ")
                print(f"      PostMerge: Candidate after merge (text='{preview}...', box={current_candidate_group.bbox})")
        else:
            if debug_this_page: print(f"      PostMerge: CANNOT merge. Finalizing current Candidate.")
            finalized_group_block = current_candidate_group.finalize()
            if finalized_group_block['lines']: 
                merged_final_blocks.append(finalized_group_block)
                if debug_this_page: print(f"        Added finalized Candidate block: lines_content='{finalized_group_block['lines'][0][:30]}...', box={finalized_group_block['box']}")
            elif debug_this_page:
                print(f"        SKIPPED finalized Candidate block as it had no lines after finalization.")

            current_candidate_group = PostMergedCandidate(next_block_to_consider)
            if debug_this_page: 
                preview = "".join(b for sublist in current_candidate_group.constituent_lines_list for b in sublist)[:30].replace("\n"," ")
                print(f"    PostMerge: Started new Candidate with Next. (text='{preview}...', box={current_candidate_group.bbox})")
            
    if current_candidate_group:
        if debug_this_page: print(f"  PostMerge: Finalizing last Candidate group. (box={current_candidate_group.bbox})")
        finalized_group_block = current_candidate_group.finalize()
        if finalized_group_block['lines']: 
            merged_final_blocks.append(finalized_group_block)
            if debug_this_page: print(f"      Added finalized block from last group: lines_content='{finalized_group_block['lines'][0][:30]}...', box={finalized_group_block['box']}")
        elif debug_this_page:
            print(f"      SKIPPED finalized block from last group as it had no lines after finalization.")
    
    if debug_this_page: print(f"  Post-process merging complete. Final block count: {len(merged_final_blocks)}")
    return merged_final_blocks

def extract_blocks_from_page_directly(page, debug_this_page=False):
    page_dict = page.get_text("dict", sort=True) 
    initial_json_blocks = []

    for pymupdf_block_idx, pymupdf_block in enumerate(page_dict["blocks"]):
        if pymupdf_block["type"] == 0 and "lines" in pymupdf_block and pymupdf_block["lines"]: 
            json_block_min_x0, json_block_min_y0, json_block_max_x1, json_block_max_y1 = \
                float('inf'), float('inf'), float('-inf'), float('-inf')
            font_sizes_for_this_json_block = []
            is_block_content_vertical = False 
            json_block_has_substantive_content = False
            block_text_content_for_heuristic = ""

            for line_obj_for_props in pymupdf_block["lines"]:
                if line_obj_for_props.get("wmode") == 1:
                    is_block_content_vertical = True
                for span_for_props in line_obj_for_props["spans"]:
                    span_text_stripped = span_for_props["text"].strip()
                    if span_text_stripped:
                        json_block_has_substantive_content = True
                        block_text_content_for_heuristic += span_for_props["text"]
                    font_sizes_for_this_json_block.append(span_for_props["size"])
                    if bool(span_for_props.get("flags", 0) & 4): 
                        is_block_content_vertical = True
                    sx0, sy0, sx1, sy1 = span_for_props["bbox"]
                    json_block_min_x0 = min(json_block_min_x0, sx0)
                    json_block_min_y0 = min(json_block_min_y0, sy0)
                    json_block_max_x1 = max(json_block_max_x1, sx1)
                    json_block_max_y1 = max(json_block_max_y1, sy1)
            
            if not json_block_has_substantive_content:
                if debug_this_page: print(f"  Skipping PyMuPDF block {pymupdf_block_idx} (initial pass) - no substantive content.")
                continue

            block_width = json_block_max_x1 - json_block_min_x0
            block_height = json_block_max_y1 - json_block_min_y0

            if not is_block_content_vertical and block_height > 0 and block_width > 0 and block_text_content_for_heuristic.strip():
                has_cjk = any(0x3040 <= ord(char) <= 0x30FF or 0x4E00 <= ord(char) <= 0x9FFF for char in block_text_content_for_heuristic.strip())
                avg_font_s = sum(font_sizes_for_this_json_block) / len(font_sizes_for_this_json_block) if font_sizes_for_this_json_block else 10.0
                num_lines_in_pymupdf_block = len(pymupdf_block.get("lines", []))
                total_stripped_chars = len("".join(block_text_content_for_heuristic.strip().split()))
                avg_chars_per_line_approx = total_stripped_chars / num_lines_in_pymupdf_block if num_lines_in_pymupdf_block > 0 else 0
                is_very_narrow_block = (block_width < avg_font_s * 2.0) 
                is_many_short_lines = (num_lines_in_pymupdf_block > 1 and avg_chars_per_line_approx < 2.5) 
                is_distinctly_taller_block = (block_height > block_width * 1.4) 

                if has_cjk and (is_very_narrow_block or is_many_short_lines or is_distinctly_taller_block):
                    if debug_this_page: 
                        text_preview = block_text_content_for_heuristic.strip()[:10]
                        reason = ["narrow_block" if is_very_narrow_block else "", "many_short_lines" if is_many_short_lines else "", "taller_block" if is_distinctly_taller_block else ""]
                        print(f"  Block {pymupdf_block_idx} (text: '{text_preview}...'): Applying heuristic for vertical. Reasons: {', '.join(filter(None, reason))}. H={block_height:.1f}, W={block_width:.1f}, avg_fs={avg_font_s:.1f}, pymu_lines={num_lines_in_pymupdf_block}, avg_chars_p_line={avg_chars_per_line_approx:.1f}")
                    is_block_content_vertical = True

            contentful_lines_to_sort = []
            for line_obj in pymupdf_block.get("lines", []):
                line_text_check = "".join(s["text"] for s in line_obj["spans"])
                if line_text_check.strip(): 
                    contentful_lines_to_sort.append(line_obj)
            
            if not contentful_lines_to_sort:
                 if debug_this_page: print(f"  Skipping PyMuPDF block {pymupdf_block_idx} (initial pass) - no contentful lines after filtering.")
                 continue

            sorted_pymupdf_lines = contentful_lines_to_sort 
            if is_block_content_vertical:
                sorted_pymupdf_lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0])) 
            else: 
                sorted_pymupdf_lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))

            texts_for_json_lines_array = []
            for pymupdf_line_obj in sorted_pymupdf_lines: 
                current_pymupdf_line_text = "".join(s["text"] for s in pymupdf_line_obj["spans"])
                texts_for_json_lines_array.append(current_pymupdf_line_text)
            
            if not texts_for_json_lines_array:
                if debug_this_page: print(f"  Warning (initial pass): PyMuPDF block {pymupdf_block_idx} produced no lines for JSON output.")
                continue

            avg_font_size = round(sum(font_sizes_for_this_json_block) / len(font_sizes_for_this_json_block), 1) if font_sizes_for_this_json_block else 10.0
            
            initial_json_blocks.append({
                "box": [int(json_block_min_x0), int(json_block_min_y0), 
                         int(json_block_max_x1), int(json_block_max_y1)],
                "vertical": is_block_content_vertical,
                "font_size": avg_font_size,
                "lines": texts_for_json_lines_array 
            })
            if debug_this_page:
                print(f"  InitialExtract: Created JSON Block (from PyMuPDF block {pymupdf_block_idx}): lines_count={len(texts_for_json_lines_array)}, bbox={initial_json_blocks[-1]['box']}, vertical={is_block_content_vertical}")
    
    initial_json_blocks.sort(key=lambda b: (
        -b["box"][0] if b["vertical"] else b["box"][1],  
        b["box"][1] if b["vertical"] else b["box"][0]    
    ))

    if debug_this_page:
        print(f"  Page {page.number}: Initial JSON blocks count: {len(initial_json_blocks)}. Sorted for merging.")
    return initial_json_blocks

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