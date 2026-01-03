"""CLI interface for py-scaffold."""

import click
from pathlib import Path
from typing import Optional

from .generator import ProjectGenerator


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗ ██╗   ██╗      ███████╗ ██████╗ █████╗ ███████╗║
║   ██╔══██╗╚██╗ ██╔╝      ██╔════╝██╔════╝██╔══██╗██╔════╝║
║   ██████╔╝ ╚████╔╝ █████╗███████╗██║     ███████║█████╗  ║
║   ██╔═══╝   ╚██╔╝  ╚════╝╚════██║██║     ██╔══██║██╔══╝  ║
║   ██║        ██║         ███████║╚██████╗██║  ██║██║     ║
║   ╚═╝        ╚═╝         ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝     ║
║                                                           ║
║        🚀 Production-Ready Python Project Generator       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""


@click.command()
@click.argument("project_name")
@click.option(
    "--template",
    "-t",
    type=click.Choice(["backend-api", "ai-project"], case_sensitive=False),
    default=None,
    help="Template to use for the project",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=".",
    help="Output directory (defaults to current directory)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force overwrite if project directory exists",
)
def main(
    project_name: str,
    template: Optional[str],
    output: str,
    force: bool,
) -> None:
    """
    Generate a production-ready Python project.

    PROJECT_NAME: Name of the project to create

    Example:
        py-scaffold my-api --template backend-api
        py-scaffold my-ai-app --template ai-project
    """
    # Display banner
    click.echo(click.style(BANNER, fg='cyan', bold=True))

    # If template is not provided, show interactive selection
    if template is None:
        while True:
            click.echo("📋 Please select a project template:\n")
            click.echo("  1. 🌐 backend-api  - Backend with FastAPI")
            click.echo("  2. 🤖 ai-project   - AI/ML project\n")

            choice = click.prompt(
                "Enter your choice",
                type=click.Choice(["1", "2"], case_sensitive=False),
                show_choices=False
            )
            
            # Map choice to template
            template = "backend-api" if choice == "1" else "ai-project"
            break
        
        click.echo()

    click.echo(click.style(f"\n✨ Creating project: {project_name}", fg='green', bold=True))
    click.echo(click.style(f"📦 Template: {template}", fg='yellow'))

    try:
        generator = ProjectGenerator(
            project_name=project_name,
            template_name=template,
            output_dir=Path(output),
            force=force,
        )

        project_path = generator.generate()

        click.echo(click.style(f"\n✅ Project created successfully at: {project_path}", fg='green', bold=True))
        click.echo(click.style("\n🚀 Next steps:", fg='cyan', bold=True))
        click.echo(f"  📁 cd {project_name}")
        click.echo(f"  🐍 python -m venv venv")
        click.echo(f"  ⚡ source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        click.echo(f"  📦 pip install -r requirements.txt")

        if template == "ai-project":
            click.echo(f"  📓 jupyter notebook  # To explore notebooks")

        click.echo(f"  ▶️  python src/main.py")

    except Exception as e:
        click.echo(click.style(f"\n❌ Error: {str(e)}", fg='red', bold=True), err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
