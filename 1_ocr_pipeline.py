"""
OCR Pipeline — Step 1
=====================
Converts scanned PDF pages to raw text.
Automatically routes documents to the best OCR engine:

  Transcript  → GPT-4o Vision (handles RTL Arabic tables perfectly)
  Course spec → PaddleOCR (fast, local, works well for English docs)

Install:
    pip install paddlepaddle paddleocr numpy pdf2image Pillow opencv-python openai python-dotenv
    brew install poppler          # macOS
    apt-get install poppler-utils # Ubuntu/Debian

Usage:
    python 1_ocr_pipeline.py                        # all PDFs in ./input_pdfs/
    python 1_ocr_pipeline.py --file path/to/doc.pdf
    python 1_ocr_pipeline.py --dpi 200
    python 1_ocr_pipeline.py --lang ar              # force language
    python 1_ocr_pipeline.py --lang en
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from pdf2image import convert_from_path

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_DIR  = Path("./input_pdfs")
OUTPUT_DIR = Path("./ocr_outputs")
DPI        = 120   # lower = less CPU/RAM; raise to 150-200 for dense text
COOLDOWN        = 5    # seconds to pause between pages to prevent overheating
FILE_COOLDOWN   = 30   # seconds to pause between files
SKIP_NON_CS     = True  # set to False to process ALL course specs regardless of CS relevance
BG_VARIANCE_THRESHOLD = 15  # above this = patterned background → preprocess

# Arabic Unicode block range
ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

# Keywords on page 1 that confirm it's a transcript (heavily Arabic)
TRANSCRIPT_KEYWORDS = [
    "السجل الأكاديمي",
    "سجل أكاديمي",
    "السجل الاكاديمي",
    "للطالب",
]

COURSE_SPEC_KEYWORDS = [
    "course specification",
    "course title",
    "course code",
    "learning outcomes",
    "bachelor",
]

# Keywords that indicate a CS-related course spec — used to skip non-CS courses
CS_KEYWORDS = [
    # English
    "computer", "computing", "programming", "software", "network",
    "database", "algorithm", "data structure", "artificial intelligence",
    "machine learning", "cybersecurity", "information technology",
    "information system", "web", "mobile", "cloud", "operating system",
    "compiler", "digital logic", "computer architecture", "csc", "cis",
    "ceng", "it ", "cs ", "swe", "sec", "net",
    # Arabic
    "حاسب", "حاسوب", "برمجة", "شبكة", "قاعدة بيانات", "خوارزمية",
    "ذكاء اصطناعي", "أمن معلومات", "تقنية معلومات", "نظم معلومات",
    "هندسة برمجيات", "تطوير", "ويب", "خدمة", "معالج",
]


# ── Image preprocessing ────────────────────────────────────────────────────────

def has_complex_background(pil_image) -> bool:
    """
    Detect if the page has a patterned/watermarked background
    by measuring pixel variance in background corner regions.
    High variance = pattern present → needs preprocessing.
    """
    try:
        import cv2
        import numpy as np
        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        # Sample bottom-left corner — usually background in transcripts
        sample = gray[h-300:h-100, 50:300]
        variance = sample.std()
        return variance > BG_VARIANCE_THRESHOLD
    except Exception:
        return False


def preprocess_image(pil_image):
    """
    Clean a scanned page for better OCR:
    1. Convert to grayscale
    2. Adaptive threshold — removes background patterns/watermarks
    3. Denoise
    Returns a PIL Image.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = np.array(pil_image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold — each region normalized independently
        # removes uneven lighting, watermarks, patterned backgrounds
        cleaned = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 15
        )

        # Light denoise to remove speckling from threshold
        cleaned = cv2.fastNlMeansDenoising(cleaned, h=10)

        # Convert back to RGB — PaddleOCR requires 3-channel image
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(cleaned_rgb)

    except ImportError:
        print("   ⚠️  opencv-python not installed — skipping preprocessing.")
        print("      pip install opencv-python")
        return pil_image
    except Exception as e:
        print(f"   ⚠️  Preprocessing failed: {e} — using original image")
        return pil_image


# ── Language detection ─────────────────────────────────────────────────────────

def detect_doc(pdf_path: Path):
    """
    Detect document type and language from page 1.
    Returns: (doc_type, lang)
      doc_type: "transcript" or "course_spec"
      lang: "ar" or "en"
    """
    print("   🔎 Detecting document language from page 1...", end="", flush=True)

    # Render just page 1 at low DPI for speed
    try:
        pages = convert_from_path(str(pdf_path), dpi=100, first_page=1, last_page=1)
    except Exception as e:
        print(f" ⚠️  Could not render page 1: {e}. Defaulting to 'arabic'.")
        return "ar"

    page1 = pages[0]

    # Quick OCR on page 1 with PaddleOCR English model for language detection
    try:
        from paddleocr import PaddleOCR
        import numpy as np

        ocr_detect = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        det_results = ocr_detect.ocr(np.array(page1), cls=False)

        sample_text = ""
        if det_results and det_results[0]:
            for line in det_results[0]:
                if line and len(line) >= 2 and line[1]:
                    sample_text += line[1][0] + " "

    except Exception:
        sample_text = ""

    # Check for transcript keywords
    for kw in TRANSCRIPT_KEYWORDS:
        if kw in sample_text:
            print(" → transcript")
            return "transcript", "ar", None

    # Measure Arabic character density
    arabic_chars = len(ARABIC_PATTERN.findall(sample_text))
    total_chars  = len(sample_text.strip())

    if total_chars == 0:
        print(" → transcript (no text detected, defaulting)")
        return "transcript", "ar", None

    arabic_ratio = arabic_chars / max(total_chars, 1)

    # Also check course specification keywords
    text_lower = sample_text.lower()
    is_english_course_spec = any(k in text_lower for k in COURSE_SPEC_KEYWORDS + ["department of"])

    if is_english_course_spec and arabic_ratio < 0.3:
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (English) | CS: {cs}")
        return "course_spec", "en", cs
    elif arabic_ratio >= 0.3:
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (Arabic) | CS: {cs}")
        return "course_spec", "ar", cs
    else:
        if any(c in pdf_path.name for c in "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"):
            print(" → transcript (Arabic filename)")
            return "transcript", "ar", None
        cs = is_cs_course_spec(pdf_path, sample_text)
        print(f" → course_spec (default) | CS: {cs}")
        return "course_spec", "en", cs


def is_cs_course_spec(pdf_path: Path, sample_text: str) -> bool:
    """
    Check if a course spec is CS-related by looking at:
    1. Course title and course code in the OCR text
    2. Course content/description keywords in the OCR text
    Does NOT rely on filename.
    Returns True if CS-related, False if not.
    """
    text_lower = sample_text.lower()

    # Check for CS keywords in the OCR text (course title, description, content)
    for kw in CS_KEYWORDS:
        if kw.lower() in text_lower:
            return True

    # Check for CS course code patterns in the text (e.g. CSC 101, CENG 201)
    import re
    cs_code_pattern = re.compile(
        r"\b(CSC|CIS|CENG|SWE|NET|SEC|CID|CSCI|INFS|ITS|ICT|COMP|INFO|CYS|IS|IT)\s*\d+",
        re.IGNORECASE
    )
    if cs_code_pattern.search(sample_text):
        return True

    return False


# ── OCR ────────────────────────────────────────────────────────────────────────

# ── GPT-4o Vision OCR (for transcripts) ───────────────────────────────────────

def ocr_page_with_gpt4o(pil_image, api_key: str) -> str:
    """
    Send a single transcript page to GPT-4o Vision for OCR.
    Handles RTL Arabic tables with complex layouts perfectly.
    """
    import base64
    import io
    from openai import OpenAI

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    client = OpenAI(api_key=api_key)

    prompt = """This is a student academic transcript (السجل الأكاديمي للطالب).
Extract ALL text exactly as it appears. Rules:
- Preserve Arabic text exactly in Arabic script
- Preserve English text exactly
- For tables: read RIGHT TO LEFT for Arabic content
- Use | to separate table columns
- Extract EVERY row from EVERY table including student info, all courses and grades
- Preserve all numbers exactly as written
- Output ONLY the transcribed text, nothing else"""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.choices[0].message.content




# ── Table formatting ───────────────────────────────────────────────────────────

def format_tables(text: str) -> str:
    """
    Heuristic: lines with multiple segments separated by 2+ spaces
    are likely table rows — format them with pipe separators.
    """
    lines = []
    for line in text.split("\n"):
        cells = re.split(r" {2,}", line.strip())
        if len(cells) >= 3:
            lines.append(" | ".join(c.strip() for c in cells if c.strip()))
        else:
            lines.append(line)
    return "\n".join(lines)



def _paddle_ocr_page(ocr, img, do_preprocess: bool = False) -> str:
    """Run PaddleOCR on a single PIL image. Used for course specs."""
    import numpy as np
    if do_preprocess:
        img = preprocess_image(img)
    result = ocr.ocr(np.array(img), cls=True)
    lines = []
    if result and result[0]:
        for line in result[0]:
            if line and len(line) >= 2 and line[1]:
                text  = line[1][0]
                score = line[1][1]
                if score > 0.5:
                    lines.append(text)
    return "\n".join(lines)


# ── Core processing ────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, forced_lang, dpi: int) -> bool:
    pdf_name = pdf_path.stem
    out_dir  = OUTPUT_DIR / pdf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📄 Processing: {pdf_path.name}")

    # Detect document type and language
    if forced_lang:
        doc_type = "transcript" if forced_lang == "ar" else "course_spec"
        lang = forced_lang
        is_cs = None  # unknown when forced
    else:
        doc_type, lang, is_cs = detect_doc(pdf_path)

    # Skip non-CS course specs entirely
    if SKIP_NON_CS and doc_type == "course_spec" and is_cs is False:
        print(f"   ⏭️  Skipping — not a CS-related course spec")
        return True  # not an error, just skipped

    print(f"   🔧 Doc type: {doc_type} | Lang: {lang}")

    # Get page count without rendering all pages into RAM
    try:
        from pdf2image.exceptions import PDFInfoNotInstalledError
        import subprocess
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True
        )
        page_count = 1  # fallback
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                page_count = int(line.split(":")[1].strip())
                break
    except Exception:
        # fallback: render all just to count, then immediately free
        try:
            pages = convert_from_path(str(pdf_path), dpi=72)
            page_count = len(pages)
            del pages
        except Exception as e:
            print(f"   ❌ Failed to read PDF: {e}")
            print("      Install poppler:  brew install poppler  |  apt-get install poppler-utils")
            return False

    print(f"   📑 {page_count} pages found")

    import time

    if doc_type == "transcript":
        # ── GPT-4o Vision path (transcripts) ───────────────────────────────
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set in .env file")
            return False

        print("   🔍 Using GPT-4o Vision for transcript OCR...")
        skipped = 0
        for i in range(page_count):
            page_file = out_dir / f"page_{i+1:03d}.txt"
            if page_file.exists():
                skipped += 1
                continue

            if skipped:
                print(f"   ⏭️  Skipped {skipped} already-done page(s)")
                skipped = 0

            print(f"   🖼️  Rendering page {i+1}/{page_count}...", end="", flush=True)
            page_img = convert_from_path(str(pdf_path), dpi=dpi,
                                         first_page=i+1, last_page=i+1)[0]
            print(f" rendered  🔍 GPT-4o Vision...", end="", flush=True)

            try:
                page_text = ocr_page_with_gpt4o(page_img, api_key)
            except Exception as e:
                print(f"\n   ❌ GPT-4o Vision failed: {e}")
                return False

            page_file.write_text(page_text, encoding="utf-8")
            del page_img
            print(f" ✅ ({len(page_text)} chars)")
            time.sleep(2)

        if skipped:
            print(f"   ⏭️  Skipped {skipped} already-done page(s)")

    else:
        # ── PaddleOCR path (course specs) ───────────────────────────────────
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("❌ PaddleOCR not installed. pip install paddlepaddle paddleocr")
            return False

        print(f"   ⏳ Loading PaddleOCR model [lang={lang}]...")
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

        print("   🔎 Checking background...", end="", flush=True)
        sample_page = convert_from_path(str(pdf_path), dpi=100, first_page=1, last_page=1)[0]
        do_preprocess = has_complex_background(sample_page)
        del sample_page
        if do_preprocess:
            print(" patterned background detected → preprocessing enabled ✅")
        else:
            print(" clean background → no preprocessing needed")

        skipped = 0
        for i in range(page_count):
            page_file = out_dir / f"page_{i+1:03d}.txt"
            if page_file.exists():
                skipped += 1
                continue

            if skipped:
                print(f"   ⏭️  Skipped {skipped} already-done page(s)")
                skipped = 0

            print(f"   🖼️  Rendering page {i+1}/{page_count}...", end="", flush=True)
            page_img = convert_from_path(str(pdf_path), dpi=dpi,
                                         first_page=i+1, last_page=i+1)[0]
            print(f" rendered  🔍 PaddleOCR...", end="", flush=True)

            page_text = _paddle_ocr_page(ocr, page_img, do_preprocess=do_preprocess)
            page_text = format_tables(page_text)
            page_file.write_text(page_text, encoding="utf-8")
            del page_img
            print(f" ✅ ({len(page_text)} chars)")
            time.sleep(COOLDOWN)

        if skipped:
            print(f"   ⏭️  Skipped {skipped} already-done page(s)")

    # Rebuild full_text.txt from all pages
    all_texts = []
    for i in range(page_count):
        pf = out_dir / f"page_{i+1:03d}.txt"
        all_texts.append(pf.read_text(encoding="utf-8") if pf.exists() else "")

    sep = "\n\n" + "=" * 60 + "\n\n"
    full_text = sep.join(f"PAGE {i+1}\n{'='*60}\n{t}" for i, t in enumerate(all_texts))
    (out_dir / "full_text.txt").write_text(full_text, encoding="utf-8")

    # Save metadata
    meta = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "page_count": page_count,
        "dpi": dpi,
        "doc_type": doc_type,
        "lang": lang,
        "forced_lang": forced_lang,
        "output_dir": str(out_dir),
        "pages": [str(out_dir / f"page_{i+1:03d}.txt") for i in range(page_count)],
        "full_text": str(out_dir / "full_text.txt"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"   ✅ Saved to: {out_dir}  [lang={lang}]")
    return True



def extract_course_code(pdf_path: Path):
    """
    Quick scan of page 1 to extract the course code (e.g. CSC 101, PHYS 201).
    Used for deduplication — if same course code seen before, skip.
    Returns course code string or None if not found.
    """
    try:
        pages = convert_from_path(str(pdf_path), dpi=72, first_page=1, last_page=1)
        from paddleocr import PaddleOCR
        import numpy as np
        ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        result = ocr.ocr(np.array(pages[0]), cls=False)
        text = ""
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    text += line[1][0] + " "
        # Look for course code pattern: 2-4 uppercase letters + space + 3-4 digits
        import re
        match = re.search(r"\b([A-Z]{2,4}\s*\d{3,4})\b", text)
        if match:
            return match.group(1).strip().upper().replace(" ", "")
    except Exception:
        pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PaddleOCR pipeline — auto-detects Arabic vs English per document"
    )
    parser.add_argument("--file",      type=str,
                        help="Process a single PDF file")
    parser.add_argument("--dpi",       type=int, default=DPI,
                        help=f"Render DPI (default: {DPI})")
    parser.add_argument("--lang",      choices=["ar", "en"],
                        help="Force OCR language (skip auto-detection)")
    parser.add_argument("--input-dir", type=str, default=str(INPUT_DIR))
    args = parser.parse_args()

    if args.file:
        pdf_paths = [Path(args.file)]
        if not pdf_paths[0].exists():
            print(f"❌ File not found: {args.file}")
            sys.exit(1)
    else:
        input_dir = Path(args.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths = sorted(input_dir.rglob("*.pdf"))  # rglob searches subfolders recursively
        if not pdf_paths:
            print(f"ℹ️  No PDFs found in {input_dir.resolve()}")
            sys.exit(0)

    print(f"🚀 OCR Pipeline  |  {len(pdf_paths)} PDF(s)  |  engine=PaddleOCR")
    if args.lang:
        print(f"   ⚠️  Language forced to: {args.lang}")

    import time
    success = 0
    seen_course_codes = set()  # track course codes to avoid duplicates

    for idx, p in enumerate(pdf_paths):
        print(f"\n{'='*60}")
        print(f"📁 File {idx+1}/{len(pdf_paths)}: {p.name}")
        print(f"{'='*60}")

        # Deduplication — check course code before processing
        # Only applies to course specs (transcripts always processed)
        if not args.lang or args.lang != "ar":
            course_code = extract_course_code(p)
            if course_code and course_code in seen_course_codes:
                print(f"   ⏭️  Duplicate course spec detected ({course_code}) — skipping")
                continue
            if course_code:
                seen_course_codes.add(course_code)

        if process_pdf(p, args.lang, args.dpi):
            success += 1

        # Cooldown between files to let CPU cool down
        if idx < len(pdf_paths) - 1:
            print(f"\n⏸️  Cooling down for {FILE_COOLDOWN}s before next file...")
            for remaining in range(FILE_COOLDOWN, 0, -5):
                print(f"   {remaining}s remaining...", end="\r")
                time.sleep(min(5, remaining))
            print("   ✅ Ready for next file        ")

    print(f"\n✅ Done! {success}/{len(pdf_paths)} PDFs processed.")
    print(f"   OCR outputs → {OUTPUT_DIR.resolve()}")
    print(f"\n▶️  Next: python 2_extraction_pipeline.py")


if __name__ == "__main__":
    main()