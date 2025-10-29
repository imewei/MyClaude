
use clap::Parser;
use anyhow::Result;
use tokio::fs;
use serde::{Serialize, Deserialize};

#[derive(Parser)]
#[command(name = "cli-tool")]
#[command(about = "A sample CLI tool", long_about = None)]
struct Cli {
    #[arg(short, long)]
    input: String,

    #[arg(short, long)]
    output: String,
}

#[derive(Serialize, Deserialize)]
struct Data {
    items: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let content = fs::read_to_string(&cli.input).await?;
    let data: Data = serde_json::from_str(&content)?;

    println!("Processing {} items", data.items.len());

    let output = serde_json::to_string_pretty(&data)?;
    fs::write(&cli.output, output).await?;

    Ok(())
}
