import io

import pytest

from demand_lens.database import connect, initialize
from demand_lens.ingestion import MAX_FILE_BYTES, UploadValidationError, validate_and_extract
from demand_lens.studio import KnowledgeOpsStudio


def test_markdown_upload_preserves_heading_paths():
    data = b"# Inventory Policy\n\nIntroduction text with enough detail.\n\n## Exceptions\n\nException details live here."
    document = validate_and_extract("inventory.md", data, "text/markdown")
    assert document.extraction_method == "structured-text"
    assert [section.section_path for section in document.sections] == [
        "Inventory Policy",
        "Inventory Policy > Exceptions",
    ]


def test_html_extraction_removes_scripts_and_preserves_headings():
    data = b"<h1>Returns</h1><p>Returns are accepted within thirty days.</p><script>steal()</script><h2>Exceptions</h2><p>Final sale is excluded.</p>"
    document = validate_and_extract("returns.html", data, "text/html")
    assert "steal" not in document.text
    assert document.sections[0].section_path == "Returns"
    assert document.sections[1].section_path == "Returns > Exceptions"


def test_invalid_extensions_and_file_signatures_are_rejected():
    with pytest.raises(UploadValidationError, match="Unsupported"):
        validate_and_extract("payload.exe", b"not an executable")
    with pytest.raises(UploadValidationError, match="signature"):
        validate_and_extract("fake.pdf", b"this is not a PDF but has enough text")
    with pytest.raises(UploadValidationError, match="MIME"):
        validate_and_extract("notes.txt", b"A valid text document with sufficient content.", "application/pdf")


def test_size_limit_is_enforced_before_extraction():
    with pytest.raises(UploadValidationError, match="10 MB"):
        validate_and_extract("large.txt", b"x" * (MAX_FILE_BYTES + 1))


def test_docx_upload_extracts_heading_and_table_when_dependency_available():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Supplier Policy", level=1)
    document.add_paragraph("This policy explains approved suppliers and procurement controls.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Supplier"
    table.rows[0].cells[1].text = "Status"
    buffer = io.BytesIO()
    document.save(buffer)
    extracted = validate_and_extract("supplier.docx", buffer.getvalue())
    assert extracted.extraction_method == "python-docx"
    assert extracted.sections[0].section_path == "Supplier Policy"
    assert "Supplier | Status" in extracted.text


def test_uploaded_chunks_retain_section_and_parent_provenance():
    document = validate_and_extract(
        "policy.md",
        b"# Main Policy\n\nThis is the primary policy content with sufficient detail.\n\n## Exception\n\nThis is a documented exception with sufficient detail.",
    )
    connection = connect()
    initialize(connection)
    studio = KnowledgeOpsStudio(connection)
    source = studio.add_uploaded_source(document)
    chunks = studio.source_chunks(source["source_id"])
    assert chunks[0]["section_path"] == "Main Policy"
    assert chunks[1]["section_path"] == "Main Policy > Exception"
    assert chunks[0]["parent_id"] != chunks[1]["parent_id"]


def test_context_packing_adds_neighbor_chunks():
    connection = connect()
    initialize(connection)
    studio = KnowledgeOpsStudio(connection)
    long_text = "\n\n".join(
        [
            "Background information " * 20,
            "The zephyr escalation rule requires immediate review. " * 12,
            "Exceptions and follow-up steps are documented here. " * 15,
        ]
    )
    source = studio.add_source("Zephyr procedure", long_text)
    outcome = studio.guarded_query("What is the zephyr escalation rule?")
    source_context = [item for item in outcome["context"] if item["source_id"] == source["source_id"]]
    assert any(item["matched"] for item in source_context)
    assert len(source_context) >= 2
