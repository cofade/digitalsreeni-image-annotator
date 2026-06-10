"""
Base class for per-tool mouse / key event handlers in ImageLabel.

Each handler owns its tool-specific temp state. ImageLabel keeps a
dispatcher that routes events to the active handler. Handlers emit
back through the ImageLabel's Phase 6 signals (see ADR-018) — they
never call into the orchestrator directly.

Plain Python objects, not QObjects: no need for their own signals,
no parent-child memory model to worry about, and unit tests can
instantiate them without a Qt event loop.
"""


class ToolHandler:
    def __init__(self, label):
        # Back-reference to the ImageLabel. Used to:
        #  - emit signals (self.label.annotationCommitted.emit(...), …)
        #  - read state via the CanvasContext (self.label._ctx.X())
        #  - write to ImageLabel.annotations (paint/eraser commit paths)
        #  - trigger a repaint (self.label.update())
        self.label = label

    # --- Mouse hooks. Each returns True if the event was consumed. ---

    def on_mouse_press(self, event, img_pt) -> bool:
        return False

    def on_mouse_move(self, event, img_pt) -> bool:
        return False

    def on_mouse_release(self, event, img_pt) -> bool:
        return False

    def on_double_click(self, event, img_pt) -> bool:
        return False

    # --- Key hooks. ImageLabel routes Enter/Escape here only after the
    # higher-priority modal branches (DINO temp, sam_points, sam_box,
    # editing polygon) have had their turn. ---

    def on_enter(self) -> bool:
        return False

    def on_escape(self) -> bool:
        return False

    # --- Painter overlay drawn after committed annotations but before
    # the size indicator. Tools render their in-progress state here. ---

    def paint_overlay(self, painter) -> None:
        return

    # --- Lifecycle. Called when the user switches away from this tool.
    # Default is no-op (matches the existing "drop state silently"
    # behaviour); commit/discard must be explicit via Enter / Escape. ---

    def deactivate(self) -> None:
        return

    # --- Unsaved-state reporting. ImageLabel.check_unsaved_changes
    # iterates handlers to decide whether to prompt the user. ---

    def has_unsaved_state(self) -> bool:
        return False

    def commit(self) -> None:
        return

    def discard(self) -> None:
        return
