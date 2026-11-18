use clap::Parser;
use frequency_analysis_limits::plotting::post::export_to_geo_and_align;
use frequency_analysis_limits::LAMA::{lama_attack, SampledSearch};
use log::{debug, info};

// Helps rayon when calling malloc
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// The path to the directory with files you want to load
    #[arg(short, long)]
    dir_path: String,
    /// If you want to save/checkpoint progress
    #[arg(short, long)]
    save: bool,

    /// Load checkpoints
    #[arg(short, long)]
    load: bool,

    /// Skip the LAMA attack (e.g. to only run --post)
    #[arg(long)]
    skip_lama: bool,

    /// The identifier for the specific function to load the file
    #[arg(short, long)]
    name: String,

    // The identifier for the specific function to load the file
    #[arg(long)]
    t: u64,

    #[arg(long, default_value = "5")]
    dim: usize,

    #[arg(long, default_value = "0")]
    padding: i64,

    #[arg(long, default_value = "uniform")]
    dist: String,

    #[arg(long)]
    post: bool,

    #[arg(long, default_value = "1.0")]
    query_percent: f64,

    /// How far (total variation distance) the true query distribution is from --dist.
    /// The attacker only knows --dist, so this is added to epsilon
    #[arg(long, default_value = "0.0")]
    dist_noise: f64,

    /// Sampled setting (--query-percent < 1): number of independent LNS restarts
    #[arg(long, default_value = "64")]
    lns_restarts: usize,

    /// Sampled setting: wall-clock budget for the LNS in hours (0 = no limit). When it runs out the
    /// restarts stop and the best databases so far are written, instead of the job hitting its time limit
    #[arg(long, default_value = "0.0")]
    lns_hours: f64,

    /// Sampled setting: max rows kept for a pair whose two records are both re-solved in one LNS step
    #[arg(long, default_value = "1000")]
    lns_max_rows: usize,

    /// Sampled setting: how many distinct databases (all pairs within epsilon) to output, lowest error first
    #[arg(long, default_value = "1000")]
    lns_outputs: usize,
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .is_test(false)
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .expect("Logger failed to init!");

    debug!("Args handed in: {:?}", args);

    debug!(
        "Rayon is operating with {} threads.",
        rayon::current_num_threads()
    );

    if !args.skip_lama {
        lama_attack(
            &args.name,
            &args.dir_path,
            &args.dist,
            &args.t,
            &args.dim,
            &args.padding,
            &args.save,
            &args.query_percent,
            &args.dist_noise,
            &SampledSearch {
                restarts: args.lns_restarts,
                lns_hours: args.lns_hours,
                max_rows_per_pair: args.lns_max_rows,
                outputs: args.lns_outputs,
            },
        );
    }

    // Do post-processing - export to GeoJSON and align
    if args.post {
        let dir = &args.dir_path;
        let remin_path = format!("{dir}/{0}/remin", args.name,);
        let less_path = format!("{dir}/{0}/even_less", args.name,);
        let unique_name = format!(
            "{0}_prob{1}.0_{2}",
            args.name, args.query_percent, args.dist
        );

        export_to_geo_and_align(
            dir,
            &remin_path,
            &less_path,
            &unique_name,
            &args.dir_path,
            &args.name,
        );
    }

    info!("Moriturus te saluto");
}
