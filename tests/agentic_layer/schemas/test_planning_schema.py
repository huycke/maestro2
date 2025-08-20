import pytest
from ai_researcher.agentic_layer.schemas.planning import ReportSection, ReportSectionBase

def test_report_section_base_is_subset_of_report_section():
    """
    Tests that the fields in ReportSectionBase are a strict subset of the fields
    in ReportSection, excluding the 'subsections' field.

    This test prevents schema drift by ensuring that if a new field is added to
    ReportSection, the developer is reminded to also add it to ReportSectionBase
    if it should be part of the LLM's schema.
    """
    # Get the set of field names from both models
    report_section_fields = set(ReportSection.model_fields.keys())
    report_section_base_fields = set(ReportSectionBase.model_fields.keys())

    # The 'subsections' field is intentionally excluded from the base model
    # to prevent circular references in the JSON schema.
    expected_diff = {'subsections'}

    # Calculate the actual difference in fields
    actual_diff = report_section_fields - report_section_base_fields

    # Assert that the only difference is the 'subsections' field
    assert actual_diff == expected_diff, (
        f"Fields in ReportSectionBase have drifted from ReportSection.\\n"
        f"Difference: {actual_diff}. Expected difference: {expected_diff}.\\n"
        f"If you added a new field to ReportSection, please also add it to "
        f"ReportSectionBase in `schemas/planning.py` to keep the LLM schema updated."
    )
