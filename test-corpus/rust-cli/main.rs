use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

#[derive(Parser, Debug)]
#[command(name = "datatool", about = "Async data processing CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Count lines in a file
    Count {
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Concatenate files into an output
    Cat {
        #[arg(short, long)]
        output: PathBuf,
        inputs: Vec<PathBuf>,
    },
}

async fn count_lines(path: &PathBuf) -> Result<usize> {
    let content = fs::read_to_string(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(content.lines().count())
}

async fn cat_files(output: &PathBuf, inputs: &[PathBuf]) -> Result<()> {
    let mut out = fs::File::create(output)
        .await
        .with_context(|| format!("creating {}", output.display()))?;
    for input in inputs {
        let bytes = fs::read(input)
            .await
            .with_context(|| format!("reading {}", input.display()))?;
        out.write_all(&bytes).await?;
    }
    out.flush().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Count { path } => {
            let n = count_lines(&path).await?;
            println!("{}: {} lines", path.display(), n);
        }
        Command::Cat { output, inputs } => {
            cat_files(&output, &inputs).await?;
            println!("wrote {} files to {}", inputs.len(), output.display());
        }
    }
    Ok(())
}
