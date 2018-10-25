extern crate rand;
use rand::Rng;
use rand::distributions::{Distribution, Exp};
use rand::prng::isaac::IsaacRng;
use rand::thread_rng;

use std::fmt;

use std::f64::INFINITY;

use std::collections::HashMap;

const EPSILON: f64 = 1e-10;
#[derive(Clone, Debug)]
struct Job {
    size: f64,
    rem_size: f64,
    arrival_time: f64,
}

impl Job {
    fn new(size: f64, arrival_time: f64) -> Self {
        Self {
            size: size,
            rem_size: size,
            arrival_time: arrival_time,
        }
    }
    fn work(&mut self, amount: f64) {
        self.rem_size -= amount;
    }
}

#[derive(Debug)]
struct Completion {
    size: f64,
    response_time: f64,
}

impl Completion {
    fn from_job(job: Job, time: f64) -> Self {
        Self {
            size: job.size,
            response_time: time - job.arrival_time,
        }
    }
}

trait Dispatch: fmt::Display {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>) -> usize;
}

#[derive(Debug)]
struct Random {
    rng: IsaacRng,
}

impl Random {
    fn new(seed: u64) -> Self {
        Self {
            rng: IsaacRng::new_from_u64(seed),
        }
    }
}

impl Dispatch for Random {
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        self.rng.gen_range(0, queues.len())
    }
}

impl fmt::Display for Random {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Random")
    }
}

struct JSQ {}

impl JSQ {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for JSQ {
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .min_by_key(|j| j.1.len())
            .unwrap()
            .0
    }
}

impl fmt::Display for JSQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JSQ")
    }
}

#[derive(Debug)]
struct JIQ {
    rng: IsaacRng,
}

impl JIQ {
    fn new(seed: u64) -> Self {
        Self {
            rng: IsaacRng::new_from_u64(seed),
        }
    }
}

impl Dispatch for JIQ {
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .position(|q| q.is_empty())
            .unwrap_or_else(|| self.rng.gen_range(0, queues.len()))
    }
}

impl fmt::Display for JIQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JIQ")
    }
}

struct LWL {}

impl LWL {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for LWL {
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| (j.0, j.1.iter().map(|j| j.rem_size).sum::<f64>()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl fmt::Display for LWL {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LWL")
    }
}
struct LWL_me {}

impl LWL_me {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for LWL_me {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .iter()
                        .map(|j| j.rem_size)
                        .map(|f| if f > job_size { 0.0 } else { f })
                        .sum::<f64>()
                        + thread_rng().gen_range(0.0, job_size * 0.0001),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl fmt::Display for LWL_me {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LWL_me")
    }
}
struct Cost {}

impl Cost {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for Cost {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .iter()
                        .map(|j| j.rem_size.min(job_size))
                        .sum::<f64>()
                        + thread_rng().gen_range(0.0, job_size * 0.0001),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}
impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cost")
    }
}
struct LWL_2me {}

impl LWL_2me {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for LWL_2me {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .iter()
                        .map(|j| j.rem_size)
                        .map(|f| if f > 2.0 * job_size { 0.0 } else { f })
                        .sum::<f64>()
                        + thread_rng().gen_range(0.0, job_size * 0.0001),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl fmt::Display for LWL_2me {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LWL_2me")
    }
}
struct IMD {
    sent: HashMap<i64, Vec<f64>>,
    c: f64,
    rng: IsaacRng,
}

impl IMD {
    fn new(c: f64) -> Self {
        Self {
            c: c,
            sent: HashMap::new(),
            rng: IsaacRng::new_from_u64(0),
        }
    }
}

impl Dispatch for IMD {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>) -> usize {
        let p = (job_size.log(self.c)).floor() as i64;
        let rng = &mut self.rng;
        let work_in_p = self.sent.entry(p).or_insert_with(|| {
            (0..queues.len())
                .map(|_| rng.gen_range(0.0, job_size * 0.001))
                .collect()
        });
        let smallest = work_in_p
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        work_in_p[smallest] += job_size;
        smallest
    }
}

impl fmt::Display for IMD {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IMD({})", self.c)
    }
}

fn simulate(
    end_time: f64,
    lambda: f64,
    size_dist: &Size,
    dispatcher: &mut impl Dispatch,
    k: usize,
    seed: u64,
) -> Vec<Completion> {
    let mut current_time: f64 = 0.;
    let mut queues: Vec<Vec<Job>> = vec![vec![]; k];
    let mut completions: Vec<Completion> = vec![];

    let arrival_generator = Exp::new(lambda);
    let mut rng = IsaacRng::new_from_u64(seed);
    let mut arrival_increment = arrival_generator.sample(&mut rng);
    while current_time < end_time
        || queues
            .iter()
            .any(|q| q.iter().any(|j| j.arrival_time < end_time))
    {
        queues
            .iter_mut()
            .for_each(|q| q.sort_by(|a, b| a.rem_size.partial_cmp(&b.rem_size).unwrap()));
        let job_increment = queues.iter().fold(INFINITY, |a, q| {
            if let Some(job) = q.get(0) {
                a.min(job.rem_size)
            } else {
                a
            }
        });
        let increment = arrival_increment.min(job_increment);
        current_time += increment;
        arrival_increment -= increment;
        let arrival_occured = arrival_increment < EPSILON;
        for queue in &mut queues {
            if !queue.is_empty() {
                queue[0].work(increment / k as f64);
                if queue[0].rem_size < EPSILON {
                    let finished_job = queue.remove(0);
                    completions.push(Completion::from_job(finished_job, current_time));
                }
            }
        }
        if arrival_occured {
            let new_size = size_dist.sample(&mut rng);
            let i = dispatcher.dispatch(new_size, &queues);
            queues[i].push(Job::new(new_size, current_time));
            arrival_increment = arrival_generator.sample(&mut rng);
        }
    }
    //Treat all jobs unfinished at end as immediately completing
    for queue in queues {
        for job in queue {
            completions.push(Completion::from_job(job, current_time));
        }
    }
    completions
}
#[derive(Clone, Debug)]
enum Size {
    Exp(f64),
    Pareto(f64),
    Hyper(f64, f64, f64),
}

impl Distribution<f64> for Size {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            &Size::Exp(lambda) => {
                let dist = Exp::new(lambda);
                dist.sample(rng)
            }
            &Size::Pareto(alpha) => rng.gen_range::<f64>(0., 1.).powf(-1. / alpha),
            &Size::Hyper(low, high, low_prob) => {
                let mean = if rng.gen_range::<f64>(0., 1.) < low_prob {
                    low
                } else {
                    high
                };
                let dist = Exp::new(1.0 / mean);
                dist.sample(rng)
            }
        }
    }
}

impl Size {
    fn balanced_hyper(covariance: f64) -> Self {
        let high = 2.0 * covariance + 1.0;
        Size::Hyper(1.0, high, high / (high + 1.0))
    }
    fn mean(&self) -> f64 {
        match self {
            &Size::Exp(lambda) => 1.0 / lambda,
            &Size::Pareto(alpha) => alpha / (alpha - 1.0),
            &Size::Hyper(low, high, low_prob) => low * low_prob + high * (1.0 - low_prob),
        }
    }
}

fn print_sim_mean(
    end_time: f64,
    rho: f64,
    size: &Size,
    dispatcher: &mut impl Dispatch,
    k: usize,
    seed: u64,
) {
    let lambda = rho / size.mean();
    let completions = simulate(end_time, lambda, size, dispatcher, k, seed);
    let mean = completions.iter().map(|c| c.response_time).sum::<f64>() / completions.len() as f64;
    println!(
        "{:?}, {}, {}, {}, {}: {}",
        size, rho, k, end_time, dispatcher, mean
    );
}
fn main() {
    let seed = 0;
    let size = Size::balanced_hyper(1000.);
    println!("Mean: {}", size.mean());
    let rho = 0.9;
    let time = 1e6;
    let k = 10;

    print_sim_mean(time, rho, &size, &mut Random::new(seed), 1, seed);
    print_sim_mean(time, rho, &size, &mut JSQ::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut JIQ::new(seed), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL_me::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL_2me::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut Cost::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(2.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.5), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.2), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.1), k, seed);
    print_sim_mean(time, rho, &size, &mut Random::new(seed), k, seed);
}
