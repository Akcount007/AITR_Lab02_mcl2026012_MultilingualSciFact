"""
Automated Layout, Overlap, and Boundary Audit for:
paper/paper.pdf
"""

import pymupdf
import os
import sys

pdf_path = os.path.join("paper", "paper.pdf")
if not os.path.exists(pdf_path):
    print(f"Error: {pdf_path} not found")
    sys.exit(1)

doc = pymupdf.open(pdf_path)
print("=== STRICT OVERLAP & BOUNDARY AUDIT ===")
print(f"Auditing file: {pdf_path} ({len(doc)} pages)\n")

total_overlaps = 0
total_margin_warnings = 0

for i, page in enumerate(doc):
    page_w = page.rect.width
    page_h = page.rect.height

    # Extract images bounding boxes
    img_infos = page.get_image_info()
    img_rects = [pymupdf.Rect(info["bbox"]) for info in img_infos]

    # Extract text blocks
    blocks = page.get_text("blocks")
    content_blocks = []
    for b in blocks:
        # Ignore running header (y < 45) and running footer (y > 750)
        if b[1] < 45 or b[3] > 750:
            continue
        content_blocks.append((pymupdf.Rect(b[:4]), b[4]))

    print(
        f"Page {i+1}: {len(img_rects)} image(s), {len(content_blocks)} text block(s)"
    )

    # 1. Margin boundaries check
    for rect, text in content_blocks:
        # 42 pt margin tolerance (standard is 45pt)
        if rect.x0 < 42 or rect.x1 > (page_w - 42):
            preview = text.strip().replace("\n", " ")[:35]
            print(
                f"  [MARGIN WARN] Text extends outside margins: x0={rect.x0:.1f}, x1={rect.x1:.1f} | '{preview}'"
            )
            total_margin_warnings += 1

    for rect in img_rects:
        if rect.x0 < 42 or rect.x1 > (page_w - 42):
            print(
                f"  [MARGIN WARN] Image extends outside margins: x0={rect.x0:.1f}, x1={rect.x1:.1f}"
            )
            total_margin_warnings += 1

    # 2. Text vs Image collision check
    for img_idx, img_r in enumerate(img_rects):
        for block_idx, (tbl_r, text) in enumerate(content_blocks):
            inter = img_r.intersect(tbl_r)
            if not inter.is_empty and inter.get_area() > 5:
                preview = text.strip().replace("\n", " ")[:45]
                print(
                    f"  [COLLISION ERROR] Img {img_idx+1} intersects Text {block_idx+1}: area={inter.get_area():.1f} pt^2 | text: '{preview}'"
                )
                total_overlaps += 1

    # 3. Text vs Text vertical collision check between DIFFERENT elements (non-adjacent y-bounds)
    for b1_idx in range(len(content_blocks)):
        for b2_idx in range(b1_idx + 1, len(content_blocks)):
            r1, t1 = content_blocks[b1_idx]
            r2, t2 = content_blocks[b2_idx]
            inter = r1.intersect(r2)
            # Only flag significant overlaps (> 30 pt^2) between separate blocks
            if not inter.is_empty and inter.get_area() > 30:
                p1 = t1.strip().replace("\n", " ")[:30]
                p2 = t2.strip().replace("\n", " ")[:30]
                # Check if it's not simply wrapped lines of the same heading
                if not (p1 in p2 or p2 in p1):
                    print(
                        f"  [TEXT OVERLAP] Block {b1_idx+1} and Block {b2_idx+1} collide! area={inter.get_area():.1f} pt^2"
                    )
                    print(f"     B1: '{p1}'")
                    print(f"     B2: '{p2}'")
                    total_overlaps += 1

print("\n" + "=" * 45)
if total_overlaps == 0 and total_margin_warnings == 0:
    print(
        "AUDIT SUCCESS: ZERO overlaps, collisions, or margin violations found!"
    )
else:
    print(
        f"AUDIT FOUND: {total_overlaps} collisions, {total_margin_warnings} margin warnings."
    )
print("=" * 45)
