"""
AI Extraction Pipeline — Step 2
================================
Reads raw OCR text from ./ocr_outputs/ and uses OpenAI to extract
structured data into JSON files in ./json_outputs/.

Automatically detects document type from page 1:
  • "Course Specification" keywords  → course_specification JSON
  • "السجل الأكاديمي للطالب" keywords → transcript JSON

Input:  ./ocr_outputs/<pdf_name>/full_text.txt
Output: ./json_outputs/<pdf_name>_course_specification.json  OR
        ./json_outputs/<pdf_name>_transcript.json

Usage:
    python 2_extraction_pipeline.py
    python 2_extraction_pipeline.py --name specific_pdf_name
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

OCR_DIR    = Path("./ocr_outputs")
OUTPUT_DIR = Path("./json_outputs")
MODEL      = "gpt-4o"
SKIP_NON_CS = True  # set to False to process ALL course specs regardless of CS relevance

# Keywords to detect document type from page 1
COURSE_SPEC_KEYWORDS = [
    "course specification",
    "course title",
    "course code",
    "course content",
    "learning outcomes",
    "clos",
]
TRANSCRIPT_KEYWORDS = [
    "السجل الأكاديمي",
    "سجل أكاديمي",
    "السجل الاكاديمي",
    "للطالب",
    "student academic record",
    "transcript",
    "academic transcript",
]

# ── JSON Schemas ───────────────────────────────────────────────────────────────

COURSE_SPEC_SCHEMA = """
Extract into this exact JSON structure. Use null for missing fields.
Return a single object (not an array).

{
  "file_code": "string (e.g. TP-153, from header/cover page)",
  "specification_year": "int(e.g.2020,2021,2022,2023,2024,etc)"
  "college": "string",
  "department": "string",
  "institution": "string",
  "course_code": "string (e.g. PHYS 101)",
  "course_title": "string",
  "is_cs_related": boolean (true if the course is related to Computer Science, IT, Software Engineering, Programming, Networks, Databases, AI, Cybersecurity, or any computing field. false otherwise),
  "general_description": "string — full paragraph description of course",
  "content_sections": [
    {
      "heading": "string (chapter/topic/unit title)",
      "topics": ["list of theoretical sub-topics, or empty array"],
      "practical_topics": ["list of practical/lab topics, or empty array"],
      "content_text": "string — ALWAYS populated, used for similarity comparison. Build it as follows: start with the heading, then append theoretical topics if any, then append practical topics prefixed with 'Practical:'. Examples: (1) heading only → 'Introduction to Computers and Programming'. (2) topics only → 'Elementary Programming: Identifiers, Variables, Assignment Statements'. (3) both → 'Elementary Programming: Identifiers, Variables. Practical: Writing simple programs'. (4) same content in both theoretical and practical → 'Elementary Programming: Identifiers, Variables. Practical: Identifiers, Variables' — include both even if identical."
    }
  ]
}

Important:
- is_cs_related must be true or false (boolean), not a string
- content_text must ALWAYS be populated — never leave it null or empty
- There are two types of course content layouts:
  TYPE 1: Practical topics are inline under each week as "Lab: ..." — extract them directly into practical_topics of that section
  TYPE 2: There is a separate "List of Practical Topics" table at the bottom of the page — you MUST extract ALL items from this separate table into practical_topics. Match them to the closest theoretical section by topic name. If a practical topic does not match any theoretical section, add it as a new section with empty topics and the practical item in practical_topics.
- For TYPE 2 documents, do NOT ignore the separate practical topics table — it is just as important as the theoretical topics table
- content_text must include BOTH theoretical and practical content prefixed with 'Practical:' even if they are identical
- Only extract the fields listed above — nothing else
- Return a single object, not wrapped in an array
"""

TRANSCRIPT_SCHEMA = """
Extract into this exact JSON structure. Use null for missing fields.
Return a single object (not an array).

{
  "student_info": {
    "student_name": "string — full name exactly as written in Arabic",
    "student_id": "string — the university/academic ID number",
    "national_id": "string or null — the civil/national ID number",
    "institution": "string or null",
    "college": "string or null",
    "major": "string or null",
    "degree": "string or null",
    "student_status": "string or null",
    "print_date": "string or null"
  },
  "summary": {
    "cumulative_gpa": "string or null — the final cumulative GPA (e.g. 4.58)",
    "total_credit_hours": "string or null — total registered hours",
    "total_points": "string or null"
  },
  "courses": [
    {
      "semester": "string — semester name exactly as written (e.g. الفصل الأول 1446ه)",
      "course_code": "string — exact course code (e.g. CSC 1102, MATH 110)",
      "course_name": "string — exact course name in Arabic",
      "credit_hours": "string — individual course credit hours (e.g. 2 or 3), NOT the semester total",
      "grade_letter": "string — letter grade (e.g. أ+, ب+, ج, هـ)",
      "grade_numeric": "string — numeric grade percentage (e.g. 95.00, 85.00)",
      "grade_points": "string — weighted grade points (e.g. 14.25, 9.00)"
    }
  ]
}

Critical extraction rules:
- Extract ALL courses from ALL semesters — do not skip any course
- credit_hours for each course is the individual course hours (2 or 3), NOT the semester total (16 or 17)
- In the table columns: course_name is rightmost, course_code is next, then credit_hours, then grade_letter, then grade_numeric, then grade_points (leftmost)
- The semester total hours row (e.g. 16.00 total) should NOT be confused with individual course credit hours
- grade_points is the small number like 9.00, 13.50, 14.25 — NOT the grade percentage
- grade_numeric is the percentage like 85.00, 92.00, 95.00
- Preserve all Arabic text exactly as written — do not translate or correct
- The student name is in the header section, not in the table
- student_id is the رقم الأكاديمي or رقم الجامعي
- national_id is the السجل المدني
"""

# ── Prompts ────────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a precise data extraction engine. You read raw OCR text 
from scanned academic documents and extract structured information into JSON format.

Rules:
- Extract ONLY what is present in the text — do not invent or guess missing data
- Use null for fields that are genuinely absent
- Preserve Arabic text exactly as it appears (do not translate)
- Preserve English text exactly as it appears
- Course codes should be exactly as written (e.g. PHYS 101, CSC 1202)
- Numbers should be parsed as numbers where the schema specifies number type
- Return ONLY valid JSON — no preamble, no explanation, no markdown code fences"""

DETECTION_PROMPT = """Look at this text from page 1 of a scanned PDF document.
Identify whether this is:
1. A "course_specification" — a course specification document
2. A "transcript" — a student academic transcript (السجل الأكاديمي للطالب)
3. "unknown" — neither of the above

Reply with ONLY one of these three words: course_specification, transcript, unknown

Page 1 text:
{page1_text}"""



# ── Helpers ────────────────────────────────────────────────────────────────────

CS_KEYWORDS = [
    "computer", "computing", "programming", "software", "network",
    "database", "algorithm", "data structure", "artificial intelligence",
    "machine learning", "cybersecurity", "information technology",
    "information system", "web", "mobile", "cloud", "operating system",
    "compiler", "digital logic", "computer architecture", "csc", "cis",
    "ceng", "it ", "cs ", "swe", "sec", "net",
    "حاسب", "حاسوب", "برمجة", "شبكة", "قاعدة بيانات", "خوارزمية",
    "ذكاء اصطناعي", "أمن معلومات", "تقنية معلومات", "نظم معلومات",
    "هندسة برمجيات", "تطوير", "ويب",
]

CS_PREFIXES = ["CSC", "CIS", "CENG", "SWE", "NET", "SEC", "CID",
               "CSCI", "INFS", "ITS", "ICT", "COMP", "INFO"]


def is_cs_related(pdf_name: str, page1_text: str) -> bool:
    """
    Check if a course spec is CS-related using course title,
    course code, and content keywords from the OCR text.
    Does NOT rely on filename.
    """
    import re
    text_lower = page1_text.lower()

    # Check for CS keywords in course title and content
    for kw in CS_KEYWORDS:
        if kw.lower() in text_lower:
            return True

    # Check for CS course code patterns in the text (e.g. CSC 101, CENG 201)
    cs_code_pattern = re.compile(
        r"\b(CSC|CIS|CENG|SWE|NET|SEC|CID|CSCI|INFS|ITS|ICT|COMP|INFO|CYS|IS|IT)\s*\d+",
        re.IGNORECASE
    )
    if cs_code_pattern.search(page1_text):
        return True

    return False


def detect_doc_type_heuristic(page1_text: str) -> str:
    text_lower = page1_text.lower()
    cs_score = sum(1 for kw in COURSE_SPEC_KEYWORDS if kw in text_lower)
    tr_score = sum(1 for kw in TRANSCRIPT_KEYWORDS if kw in page1_text)
    if cs_score >= 2:
        return "course_specification"
    if tr_score >= 1:
        return "transcript"
    return "unknown"


def detect_doc_type_with_llm(client, page1_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": DETECTION_PROMPT.format(page1_text=page1_text[:2000])
            }]
        )
        result = response.choices[0].message.content.strip().lower()
        if result in ("course_specification", "transcript"):
            return result
        return "unknown"
    except Exception as e:
        print(f"   ⚠️  LLM detection failed: {e}")
        return "unknown"


def extract_structured_data(client, full_text: str, doc_type: str) -> dict:
    if doc_type == "course_specification":
        schema = COURSE_SPEC_SCHEMA
        instruction = """Extract all course specification data from this OCR text into the JSON schema provided.

CRITICAL RULES FOR content_sections:
- Read the ENTIRE OCR text carefully — do not stop early
- Look for TWO types of content tables:
  TYPE 1: A single table where each row has a topic AND a "Lab:" practical inline — extract directly
  TYPE 2: TWO separate tables — one for theoretical topics, one titled "List of Topics (Practical Aspects)" or similar
- For TYPE 2 documents:
  * Extract ALL rows from the theoretical table as headings
  * Extract ALL rows from the practical table
  * Try to match each practical to the closest theoretical topic by subject
  * If a practical item does NOT clearly match any theoretical topic, add it as a NEW section with empty topics[] and the practical item in practical_topics[]
  * Do NOT skip any practical item — every row from the practical table must appear somewhere
- content_text MUST always be populated — combine heading + topics + "Practical: " + practical_topics
- If practical_topics exist, content_text MUST include "Practical: ..." at the end
- Do NOT leave content_sections empty or incomplete — extract ALL rows from ALL tables
- The number of items in practical_topics across all sections must equal the total rows in the practical table"""
    else:
        schema = TRANSCRIPT_SCHEMA
        instruction = "Extract all student transcript data from this OCR text into the JSON schema provided."

    prompt = f"""{instruction}

JSON Schema to populate:
{schema}

--- RAW OCR TEXT START ---
{full_text}
--- RAW OCR TEXT END ---

Return ONLY the populated JSON object. No markdown, no explanation."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=16000,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present despite response_format
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  JSON parse error attempt {attempt+1}: {e}. Retrying...")
                time.sleep(3)
            else:
                print(f"   ❌ Could not parse JSON after {max_retries} attempts")
                return {"_extraction_error": str(e), "_raw_response": raw}

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   ⚠️  API error attempt {attempt+1}: {e}. Retrying in 5s...")
                time.sleep(5)
            else:
                return {"_api_error": str(e)}


def check_if_cs_related(client, page1_text: str) -> bool:
    """
    Quick check if a course spec is CS-related using a small prompt.
    Returns True if CS-related, False otherwise.
    CS fields include: Computer Science, IT, Software Engineering,
    Programming, Networks, Databases, AI, Cybersecurity, computing.
    """
    # Fast heuristic first — check course title/code keywords
    cs_keywords = [
        "csc", "cis", "cs ", "software", "programming", "network",
        "database", "artificial intelligence", "cybersecurity", "computing",
        "information technology", "computer", "data structure", "algorithm",
        "operating system", "web ", "mobile", "cloud", "machine learning",
    ]
    text_lower = page1_text.lower()
    if any(kw in text_lower for kw in cs_keywords):
        return True

    # If heuristic is inconclusive, ask the LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": f"""Is this course related to Computer Science, IT, Programming, Networks, Databases, AI, or Cybersecurity?
Reply with only: yes or no

Course info:
{page1_text[:500]}"""
            }]
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception:
        return True  # default to processing if check fails



def process_ocr_output(ocr_dir: Path, client) -> bool:
    pdf_name = ocr_dir.name
    full_text_file = ocr_dir / "full_text.txt"
    page1_file     = ocr_dir / "page_001.txt"

    if not full_text_file.exists():
        print(f"   ⚠️  No full_text.txt in {ocr_dir}")
        return False

    print(f"\n📂 Processing: {pdf_name}")

    full_text  = full_text_file.read_text(encoding="utf-8")
    page1_text = page1_file.read_text(encoding="utf-8") if page1_file.exists() else full_text[:1000]

    # Detect document type
    print("   🔎 Detecting document type...", end="", flush=True)
    doc_type = detect_doc_type_heuristic(page1_text)
    if doc_type == "unknown":
        doc_type = detect_doc_type_with_llm(client, page1_text)
    print(f" → {doc_type}")

    if doc_type == "unknown":
        print("   ⚠️  Could not determine document type. Saving as 'unknown'.")

    # Skip non-CS course specs
    if SKIP_NON_CS and doc_type == "course_specification":
        if not is_cs_related(pdf_name, page1_text):
            print("   ⏭️  Skipping — not a CS-related course spec")
            return True


    # Extract structured data via OpenAI
    print("   🤖 Extracting structured data...", end="", flush=True)
    data = extract_structured_data(client, full_text, doc_type)
    print(f" ✅ ({len(json.dumps(data))} chars)")

    data["_metadata"] = {
        "source_pdf_name": pdf_name,
        "document_type": doc_type,
        "ocr_output_dir": str(ocr_dir),
        "extracted_by": "2_extraction_pipeline.py",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{pdf_name}_{doc_type}.json"
    out_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   💾 Saved: {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="AI Extraction Pipeline — OCR text → structured JSON via OpenAI")
    parser.add_argument("--dir",  type=str, default=str(OCR_DIR))
    parser.add_argument("--name", type=str, help="Process only this OCR folder name")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found.")
        print("   Add OPENAI_API_KEY=sk-... to your .env file")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    ocr_root = Path(args.dir)
    if not ocr_root.exists():
        print(f"❌ OCR directory not found: {ocr_root}")
        sys.exit(1)

    if args.name:
        dirs = [ocr_root / args.name]
        if not dirs[0].exists():
            print(f"❌ Not found: {dirs[0]}")
            sys.exit(1)
    else:
        dirs = [d for d in sorted(ocr_root.iterdir()) if d.is_dir()]
        if not dirs:
            print(f"ℹ️  No OCR output directories found in {ocr_root}")
            print("   Run step 1 first: python 1_ocr_pipeline.py")
            sys.exit(0)

    print(f"🚀 Extraction Pipeline — {len(dirs)} document(s) to process")

    success = 0
    seen_course_codes = set()  # track course codes to avoid duplicate extractions

    for d in dirs:
        # Deduplication — check if we already extracted this course code
        meta_file = d / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("doc_type") == "course_spec":
                    # Try to find course code from existing JSON output
                    json_outputs = list(OUTPUT_DIR.glob(f"{d.name}_*.json"))
                    for jf in json_outputs:
                        try:
                            jdata = json.loads(jf.read_text(encoding="utf-8"))
                            course_code = jdata.get("course_code", "").strip().upper().replace(" ", "")
                            if course_code and course_code in seen_course_codes:
                                print(f"\n📂 {d.name}")
                                print(f"   ⏭️  Duplicate course spec ({course_code}) — skipping extraction")
                                success += 1
                                break
                            if course_code:
                                seen_course_codes.add(course_code)
                        except Exception:
                            pass
                    else:
                        if process_ocr_output(d, client):
                            success += 1
                    continue
            except Exception:
                pass

        if process_ocr_output(d, client):
            success += 1

    print(f"\n✅ Done! {success}/{len(dirs)} documents extracted.")
    print(f"   JSON outputs → {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()