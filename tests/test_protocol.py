from uuid import UUID

from latentslate_engine.protocol import (
    ChoiceOption,
    InputType,
    InputUi,
    MediaType,
    ToolDescriptor,
    ToolInput,
    ToolOutput,
    ToolRequirement,
    WorkflowKind,
)
from latentslate_engine.tools.h3 import H3FirstLastFrameTool, H3TextToVideoTool


TOOL_ID = UUID("f51cb180-7570-428a-a173-4e0b060437ef")


def _descriptor(*, presentation_suffix: str = "", required: bool = True) -> ToolDescriptor:
    return ToolDescriptor(
        id=TOOL_ID,
        key="test.example",
        schema_revision=1,
        name=f"Example{presentation_suffix}",
        description=f"Description{presentation_suffix}",
        workflow_kind=WorkflowKind.TEXT_TO_VIDEO,
        output=ToolOutput(type=MediaType.VIDEO),
        inputs=[
            ToolInput(
                key="quality",
                label=f"Quality{presentation_suffix}",
                type=InputType.CHOICE,
                required=required,
                default="draft",
                options=[
                    ChoiceOption(
                        value="draft",
                        label=f"Draft{presentation_suffix}",
                        description=f"Fast{presentation_suffix}",
                    )
                ],
                ui=InputUi(
                    group=f"Output{presentation_suffix}",
                    advanced=bool(presentation_suffix),
                    multiline=bool(presentation_suffix),
                    placeholder=f"Choose{presentation_suffix}",
                    min=0,
                    max=10,
                    step=1,
                    unit=f"steps{presentation_suffix}",
                ),
            )
        ],
        requirements=[ToolRequirement(bundle_id="example")],
    ).with_schema_hash()


def test_schema_hash_ignores_presentation_copy() -> None:
    assert _descriptor().schema_hash == _descriptor(presentation_suffix=" revised").schema_hash


def test_schema_hash_changes_with_machine_contract() -> None:
    assert _descriptor(required=True).schema_hash != _descriptor(required=False).schema_hash


def test_h3_names_follow_latentslate_workflow_taxonomy() -> None:
    text = H3TextToVideoTool().descriptor
    first_last = H3FirstLastFrameTool().descriptor

    assert text.name == "Text to Video"
    assert first_last.name == "First/Last Frame Video"
    assert text.inputs[0].label == "Prompt"
    assert first_last.inputs[0].label == "Prompt"
