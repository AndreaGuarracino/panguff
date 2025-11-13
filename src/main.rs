use clap::{Parser, Subcommand};
use log::{debug, info};
use rustc_hash::FxHashSet;

mod nearest;

/// Pangenome stuff - A toolkit for pangenome graph analysis
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,

    /// Verbosity level (0=error, 1=info, 2=debug)
    #[arg(short, long, global = true, default_value = "1")]
    verbose: u8,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Find nearest reference node for each graph node
    Nearest {
        /// GFA pangenome graph file
        #[arg(short, long)]
        gfa: String,

        /// Reference sequence name (can be specified multiple times)
        #[arg(short, long, conflicts_with = "reference_list")]
        reference: Vec<String>,

        /// File containing reference sequence names (one per line)
        #[arg(long, conflicts_with = "reference")]
        reference_list: Option<String>,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<String>,

        /// Number of threads for parallel processing
        #[arg(short, long, default_value = "4")]
        threads: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Set log level based on verbosity
    env_logger::Builder::new()
        .filter_level(match args.verbose {
            0 => log::LevelFilter::Error,
            1 => log::LevelFilter::Info,
            _ => log::LevelFilter::Debug,
        })
        .init();

    match args.command {
        Command::Nearest {
            gfa,
            reference,
            reference_list,
            output,
            threads,
        } => {
            // Configure rayon thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap();

            debug!("GFA file: {}", gfa);

            // Load reference sequences from either -r or --reference-list
            let ref_paths = if !reference.is_empty() {
                reference
            } else if let Some(ref list_file) = reference_list {
                
                nearest::load_ref_paths(list_file)?
            } else {
                return Err("Either --reference or --reference-list must be specified".into());
            };
            info!("Reference sequences: {}", ref_paths.len());

            if let Some(ref out) = output {
                debug!("Output file: {}", out);
            } else {
                debug!("Output: stdout");
            }
            debug!("Threads: {}", threads);

            let graph = nearest::Graph::from_gfa(&gfa)?;
            let ref_path_set: FxHashSet<String> = ref_paths.into_iter().collect();

            let result = nearest::find_nearest(&graph, &ref_path_set);
            let ref_positions = nearest::get_ref_positions(&graph, &ref_path_set);

            nearest::write_output(output.as_deref(), &graph, &result, &ref_positions)?;
            if let Some(ref out) = output {
                info!("Done! Results written to {}", out);
            } else {
                info!("Done!");
            }
        }
    }

    Ok(())
}
