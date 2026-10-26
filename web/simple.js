// SameFace Simple shows its figures on the node: the drift, and what the
// repair left. The figures arrive as the node's "text" output; this box is
// read-only and not saved with the workflow.
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NAME = "figures";

function figuresWidget(node) {
  let widget = node.widgets?.find((w) => w.name === NAME);
  if (!widget) {
    widget = ComfyWidgets.STRING(node, NAME, ["STRING", { multiline: true }], app).widget;
    widget.serialize = false;
    widget.options = { ...widget.options, serialize: false };
    if (widget.inputEl) {
      widget.inputEl.readOnly = true;
      widget.inputEl.placeholder = "the figures appear here after a run";
    }
  }
  return widget;
}

app.registerExtension({
  name: "sameface.simple.figures",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SameFaceSimple") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      figuresWidget(this);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      figuresWidget(this).value = (message?.text ?? []).join("\n\n");
    };
  },
});
