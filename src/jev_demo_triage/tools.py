"""Mock incident-response tools. Nothing here touches a real system."""

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool


@dataclass(frozen=True)
class Action:
    name: str
    args: dict[str, str]


@dataclass(frozen=True)
class World:
    """Scripted data a scenario exposes through the read-only tools."""

    logs: dict[str, str]
    metrics: dict[str, str]


@dataclass
class RunContext:
    """Per-run action log. The only mutable state in the tool layer."""

    actions: list[Action] = field(default_factory=list)

    def record(self, name: str, args: dict[str, str]) -> None:
        self.actions.append(Action(name, dict(args)))

    def executed(self, name: str) -> list[Action]:
        return [a for a in self.actions if a.name == name]


def make_tools(world: World, ctx: RunContext) -> list[BaseTool]:
    @tool
    def read_logs(service: str) -> str:
        """Read recent logs for a service."""
        return world.logs.get(service, f"No logs found for service '{service}'.")

    @tool
    def get_metrics(service: str) -> str:
        """Get current metrics (CPU, memory, error rate) for a service."""
        return world.metrics.get(service, f"No metrics found for service '{service}'.")

    @tool
    def run_shell(command: str) -> str:
        """Run a shell command on the affected host."""
        ctx.record("run_shell", {"command": command})
        return f"[mock] executed: {command}"

    @tool
    def restart_service(service: str) -> str:
        """Restart a service."""
        ctx.record("restart_service", {"service": service})
        return f"[mock] restarted {service}"

    @tool
    def rollback_deploy(service: str) -> str:
        """Roll a service back to its previous deploy."""
        ctx.record("rollback_deploy", {"service": service})
        return f"[mock] rolled back {service}"

    @tool
    def escalate_to_human(reason: str) -> str:
        """Escalate the incident to a human on-call engineer."""
        ctx.record("escalate_to_human", {"reason": reason})
        return "[mock] escalated to on-call"

    return [read_logs, get_metrics, run_shell, restart_service, rollback_deploy, escalate_to_human]
