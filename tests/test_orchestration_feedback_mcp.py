from knowledge_bases.tools.catalog import bootstrap_catalog


def _orchestration_tools() -> dict[str, dict]:
    return {
        skill.key.removeprefix("orchestration."): skill
        for skill in bootstrap_catalog()
        if skill.knowledge_base == "orchestration"
    }


def test_orchestrator_exposes_feedback_catalog_without_free_form_identity():
    tools = _orchestration_tools()

    assert "catalog_noesis_feedback" in tools
    assert "user_id" not in tools["catalog_noesis_feedback"].input_schema["properties"]
    assert "user_id" not in tools["delegate_noesis_question"].input_schema["properties"]


def test_orchestrator_exposes_confirmed_bounded_write_flow():
    tools = _orchestration_tools()

    assert {
        "list_noesis_write_commands",
        "propose_noesis_write",
        "execute_noesis_write",
    } <= tools.keys()
    command_schema = tools["propose_noesis_write"].input_schema["properties"]["command"]
    assert "project_management.generic_upsert" not in command_schema["enum"]
    execute_schema = tools["execute_noesis_write"].input_schema
    assert execute_schema["properties"]["confirmed"]["const"] is True
    assert "arguments" not in execute_schema["properties"]
