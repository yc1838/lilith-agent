import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.align import Align
from rich.live import Live
from rich.spinner import Spinner
from rich.theme import Theme
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from lilith_agent.config import Config
from lilith_agent.app import build_react_agent
from lilith_agent.observability import setup_logging, setup_arize, JsonlTraceCallback

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "lilith_primary": "italic magenta"
})

console = Console(theme=custom_theme)

prompt_style = Style.from_dict({
    'prompt': 'ansimagenta bold',
})

LILITH_LOGO = r"""
[magenta]██╗     ██╗██╗     ██╗████████╗██╗  ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/magenta]
[magenta]██║     ██║██║     ██║╚══██╔══╝██║  ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/magenta]
[bright_magenta]██║     ██║██║     ██║   ██║   ███████║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   [/bright_magenta]
[bright_magenta]██║     ██║██║     ██║   ██║   ██╔══██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   [/bright_magenta]
[magenta]███████╗██║███████╗██║   ██║   ██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   [/magenta]
[magenta]╚══════╝╚═╝╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   [/magenta]
           [cyan italic]🦋  ReAct Research Assistant  🦋[/cyan italic]
"""

def print_logo():
    console.print(LILITH_LOGO)

def _extract_text(content) -> str:
    """Flatten AIMessage.content to a string. Anthropic returns a list of
    content blocks (e.g. [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]);
    other providers return a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content) if content is not None else ""


_CAVEMAN_HINTS = {
    "brief": "make model talk very brief",
    "full": "make model talk like caveman (terse, substance only)",
    "ultra": "make model talk ultra-compressed",
}


def _print_caveman_status(cfg):
    if cfg.caveman:
        hint = _CAVEMAN_HINTS.get(cfg.caveman_mode, "terse mode")
        console.print(f"[italic magenta]caveman: on[/italic magenta] [dim](mode: {cfg.caveman_mode} — {hint})[/dim]")
    else:
        console.print("[dim]caveman: off[/dim]")


def main_loop(cfg):
    print_logo()
    _print_caveman_status(cfg)
    log_path = setup_logging(".lilith")
    console.print(f"[dim cyan]Logging to {log_path}[/dim cyan]")
    if setup_arize(project_name="lilith"):
        console.print("[dim cyan]Arize tracing: enabled[/dim cyan]")
    console.print("\n[dim cyan]Initializing agent...[/dim cyan]")

    try:
        graph = build_react_agent(cfg)
    except Exception as e:
        console.print(f"[bold red]Failed to build graph: {e}[/bold red]")
        sys.exit(1)

    console.print("[dim cyan]Agent ready. Type 'exit' or 'quit' to close.[/dim cyan]\n")

    session = PromptSession(style=prompt_style)

    # Persistent thread ID plus JSONL trace of every tool/LLM event for this session.
    thread_id = str(uuid.uuid4())
    trace_path = log_path.with_name(log_path.stem + ".jsonl")
    trace_cb = JsonlTraceCallback(trace_path)
    thread_config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [trace_cb],
    }
    console.print(f"[dim cyan]Trace: {trace_path}[/dim cyan]\n")
    
    while True:
        try:
            user_input = session.prompt("lilith 🦋 > ")
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
            
        text = user_input.strip()
        if not text:
            continue
            
        if text.lower() in ("exit", "quit"):
            console.print("[magenta]Goodbye! 🦋[/magenta]")
            break
            
        if text.lower() == "/clear":
            thread_id = str(uuid.uuid4())
            trace_path = log_path.with_name(f"{log_path.stem}-{thread_id[:8]}.jsonl")
            trace_cb = JsonlTraceCallback(trace_path)
            thread_config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [trace_cb],
            }
            console.print("[dim cyan]Conversation memory cleared. Starting a new thread.[/dim cyan]\n")
            continue
            
        if text.lower().startswith("/caveman") or text.lower().startswith("/cavemen"):
            parts = text.split()
            if len(parts) > 1:
                arg = parts[1].lower()
                if arg in ("off", "false", "no", "disable"):
                    cfg.caveman = False
                elif arg in ("on", "true", "yes", "enable"):
                    cfg.caveman = True
                else:
                    cfg.caveman = True
                    cfg.caveman_mode = arg
            else:
                cfg.caveman = not cfg.caveman

            state_str = "ENABLED" if cfg.caveman else "DISABLED"
            console.print(f"[dim cyan]CAVEMAN MODE {state_str} (mode: {cfg.caveman_mode})[/dim cyan]\n")
            continue
            
        # ReAct agent uses "messages" state natively
        input_state = {"messages": [("user", text)], "iterations": 0}
        
        console.print("\n")
        
        with Live(Spinner("dots", text="[magenta]Thinking...[/magenta]"), refresh_per_second=10) as live:
            last_message = None
            try:
                for chunk in graph.stream(input_state, thread_config, stream_mode="values"):
                    if "messages" in chunk:
                        # stream_mode="values" returns the full state after each node.
                        # the last message added is the newest state.
                        latest = chunk["messages"][-1]
                        
                        if latest.type == "ai" and latest.tool_calls:
                            tool_strs = []
                            for tc in latest.tool_calls:
                                name = tc.get("name", "unknown")
                                dict_args = tc.get("args", {})
                                if isinstance(dict_args, dict):
                                    arg_str = ", ".join(f"{k}={repr(v)[:50] + '...' if len(repr(v)) > 50 else repr(v)}" for k, v in dict_args.items())
                                else:
                                    arg_str = repr(dict_args)[:50] + '...' if len(repr(dict_args)) > 50 else repr(dict_args)
                                tool_strs.append(f"{name}({arg_str})")
                            tools = " | ".join(tool_strs)
                            live.console.print(f"[dim cyan] [TOOL][/dim cyan] {tools}")
                            
                        elif latest.type == "tool":
                            content_str = str(latest.content).replace('\n', ' ')
                            if len(content_str) > 300:
                                content_preview = content_str[:150] + " ... " + content_str[-150:]
                            else:
                                content_preview = content_str
                            live.console.print(f"[dim cyan] [OBSERVATION][/dim cyan] {latest.name}: {content_preview}")
                            
                        last_message = latest
                        
            except Exception as e:
                live.console.print(f"[bold red]Agent Error: {e}[/bold red]")
                import traceback
                traceback.print_exc()
                continue
                
        # Final output
        if last_message and last_message.type == "ai":
            answer = _extract_text(last_message.content)
            if answer:
                console.print(Panel(Markdown(answer), title="🦋 [magenta]Lilith's Answer[/magenta]", border_style="magenta"))
            else:
                console.print("[yellow]Agent finished but returned no text content.[/yellow]\n")
        else:
             console.print("[yellow]Agent execution ended.[/yellow]\n")

def main():
    cfg = Config.from_env()
    main_loop(cfg)

if __name__ == "__main__":
    main()
