"""CLI interface for py-scaffold."""

import click
from pathlib import Path
from typing import Optional

from .generator import ProjectGenerator


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
    # If template is not provided, show interactive selection
    if template is None:
        while True:
            click.echo("Please select a project template:\n")
            click.echo("1. backend-api  - Backend with FastAPI")
            click.echo("2. ai-project   - AI/ML project (Up coming)\n")

            choice = click.prompt(
                "Enter your choice",
                type=click.Choice(["1", "2"], case_sensitive=False),
                show_choices=False
            )
            
            if choice == "2":
                click.echo("\nThe 'ai-project' template is coming soon! Please select 'backend-api' for now.\n")
                continue  # Go back to the start of the loop
            
            # If we get here, choice must be "1"
            template = "backend-api"
            break
        
        click.echo()

    click.echo(f"Creating project: {project_name}")
    click.echo(f"Template: {template}")

    try:
        generator = ProjectGenerator(
            project_name=project_name,
            template_name=template,
            output_dir=Path(output),
            force=force,
        )

        project_path = generator.generate()

        click.echo(f"\nProject created successfully at: {project_path}")
        click.echo("\nNext steps:")
        click.echo(f"  cd {project_name}")
        click.echo("  python -m venv venv")
        click.echo("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        click.echo("  pip install -r requirements.txt")

        if template == "ai-project":
            click.echo("  jupyter notebook  # To explore notebooks")

        click.echo("  python src/main.py")

    except Exception as e:
        click.echo(f"\nError: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
