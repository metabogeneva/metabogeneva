from time import monotonic


class NullMonitor:
    def state(self, text: str) -> None:
        pass

    def log(self, text: str) -> None:
        pass


class TerminalMonitor(NullMonitor):
    def state(self, text: str) -> None:
        pass

    def log(self, text: str) -> None:
        print(text)


class IPythonMonitor(NullMonitor):
    def __init__(self) -> None:
        from IPython.display import display
        from ipywidgets import Layout, Text, Textarea, VBox

        self.state_field = Text(layout=Layout(width="75%"))
        self.log_field = Textarea(layout=Layout(width="75%", height="24ex"))
        display(VBox([self.state_field, self.log_field]))
        self.last_state_time = 0.0

    def state(self, text: str) -> None:
        now = monotonic()
        if text and now - self.last_state_time < 0.1:
            return
        self.last_state_time = now
        self.state_field.value = text

    def log(self, text: str) -> None:
        self.log_field.value = self.log_field.value + text + "\n"
