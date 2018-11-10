extern crate rand;
use rand::Rng;
use rand::distributions::{Distribution, Exp};
use rand::prng::isaac::IsaacRng;
use rand::thread_rng;

extern crate quadrature;
use quadrature::integrate;

extern crate noisy_float;
use noisy_float::prelude::*;

use std::fmt;

use std::f64::INFINITY;

use std::collections::HashMap;
use std::collections::BTreeMap;

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
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize;
}

impl<S: Dispatch + ?Sized> Dispatch for Box<S> {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        (**self).dispatch(job_size, queues)
    }
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
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
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
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
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
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
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
    fn dispatch(&mut self, _job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| (j.0, j.1.values().map(|j| j.rem_size).sum::<f64>()))
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
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .values()
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
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1.values().map(|j| j.rem_size.min(job_size)).sum::<f64>()
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
struct Cost2 {
    size: Size,
    lambda: f64,
}

impl Cost2 {
    fn new(size_dist: &Size, lambda: f64) -> Self {
        Self {
            size: size_dist.clone(),
            lambda: lambda,
        }
    }
}

impl Dispatch for Cost2 {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .values()
                        .map(|j| {
                            let small_size = j.rem_size.min(job_size);
                            let large_size = j.rem_size.max(job_size);
                            let normal_rho = self.size.mean_given_below(job_size) * self.lambda;
                            let large_rho = self.size.mean_given_below(large_size) * self.lambda;
                            let lambda_above_large =
                                (1.0 - self.size.fraction_below(large_size)) * self.lambda;
                            (small_size / (1.0 - normal_rho))
                                * (1.0 + lambda_above_large * large_size / (1.0 - large_rho))
                        })
                        .sum::<f64>()
                        + thread_rng().gen_range(0.0, job_size * 0.0001),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}
impl fmt::Display for Cost2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cost2")
    }
}
struct Cost3 {
    size: Size,
    lambda: f64,
}

impl Cost3 {
    fn new(size_dist: &Size, lambda: f64) -> Self {
        Self {
            size: size_dist.clone(),
            lambda: lambda,
        }
    }
}

impl Dispatch for Cost3 {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .values()
                        .map(|j| {
                            let small_size = j.rem_size.min(job_size);
                            let large_size = j.rem_size.max(job_size);
                            let normal_rho = self.size.mean_given_below(job_size) * self.lambda;
                            //let large_rho = self.size.mean_given_below(large_size) * self.lambda;
                            let large_desceding_bp =
                                self.size.descending_busy_period(large_size, self.lambda);
                            let lambda_above_large =
                                (1.0 - self.size.fraction_below(large_size)) * self.lambda;
                            (small_size / (1.0 - normal_rho))
                                * (1.0 + lambda_above_large * large_size * large_desceding_bp)
                        })
                        .sum::<f64>()
                        + thread_rng().gen_range(0.0, job_size * 0.0001),
                )
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}
impl fmt::Display for Cost3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cost3")
    }
}
struct LWL_2me {}

impl LWL_2me {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for LWL_2me {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        queues
            .iter()
            .enumerate()
            .map(|j| {
                (
                    j.0,
                    j.1
                        .values()
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
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
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
struct IMDbelow {
    sent: HashMap<i64, Vec<f64>>,
    c: f64,
    rng: IsaacRng,
}

impl IMDbelow {
    fn new(c: f64) -> Self {
        Self {
            c: c,
            sent: HashMap::new(),
            rng: IsaacRng::new_from_u64(0),
        }
    }
}

impl Dispatch for IMDbelow {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        let p = (job_size.log(self.c)).floor() as i64;
        if !self.sent.contains_key(&p) {
            let rng = &mut self.rng;
            let table = (0..queues.len())
                .map(|_| rng.gen_range(0.0, job_size * 0.001))
                .collect();
            self.sent.insert(p, table);
        }
        let mut sums_below = vec![0.0; queues.len()];
        for (&k, v) in &self.sent {
            if k <= p {
                for (i, amount_sent) in v.iter().enumerate() {
                    sums_below[i] += amount_sent;
                }
            }
        }
        let smallest = sums_below
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        self.sent.get_mut(&p).unwrap()[smallest] += job_size;
        smallest
    }
}

impl fmt::Display for IMDbelow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IMDbelow({})", self.c)
    }
}
// Load balanced SITA
struct SITA {
    size: Size,
}

impl SITA {
    fn new(size: &Size) -> Self {
        Self {
            size: size.clone(),
        }
    }
}

impl Dispatch for SITA {
    fn dispatch(&mut self, job_size: f64, queues: &Vec<BTreeMap<N64, Job>>) -> usize {
        let mean_below = self.size.mean_given_below(job_size);
        let load_below = mean_below / self.size.mean();
        (load_below * queues.len() as f64).floor() as usize
    }
}

impl fmt::Display for SITA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SITA")
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
    let mut queues: Vec<BTreeMap<N64, Job>> = vec![BTreeMap::new(); k];
    let mut completions: Vec<Completion> = vec![];

    let arrival_generator = Exp::new(lambda);
    let mut rng = IsaacRng::new_from_u64(seed);
    let mut arrival_increment = arrival_generator.sample(&mut rng);

    while current_time < end_time
        || queues
            .iter()
            .any(|q| q.values().any(|j| j.arrival_time < end_time))
    {
        let job_increment = queues.iter().fold(INFINITY, |a, q| {
            if let Some(job) = q.values().nth(0) {
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
                let min_size = *queue.keys().nth(0).unwrap();
                let mut min_size_job = queue.remove(&min_size).unwrap();
                min_size_job.work(increment / k as f64);
                if min_size_job.rem_size < EPSILON {
                    completions.push(Completion::from_job(min_size_job, current_time));
                } else {
                    let was_there = queue.insert(n64(min_size_job.rem_size), min_size_job);
                    assert!(was_there.is_none());
                }
            }
        }
        if arrival_occured {
            let new_size = size_dist.sample(&mut rng);
            let i = dispatcher.dispatch(new_size, &queues);
            let was_there = queues[i].insert(n64(new_size), Job::new(new_size, current_time));
            assert!(was_there.is_none());
            arrival_increment = arrival_generator.sample(&mut rng);
        }
    }
    //Treat all jobs unfinished at end as immediately completing
    for queue in queues {
        for job in queue.values() {
            completions.push(Completion::from_job(job.clone(), current_time));
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
        let high = 2.0 * covariance;
        Size::Hyper(1.0, high, high / (high + 1.0))
    }
    fn mean(&self) -> f64 {
        match self {
            &Size::Exp(lambda) => 1.0 / lambda,
            &Size::Pareto(alpha) => alpha / (alpha - 1.0),
            &Size::Hyper(low, high, low_prob) => low * low_prob + high * (1.0 - low_prob),
        }
    }
    fn variance(&self) -> f64 {
        match self {
            &Size::Exp(lambda) => 1.0 / lambda.powi(2),
            &Size::Pareto(alpha) => (1.0 / (alpha - 1.0)).powi(2) * alpha / (alpha - 2.0),
            &Size::Hyper(low, high, low_prob) => {
                2.0 * low * low * low_prob + 2.0 * high * high * (1.0 - low_prob)
                    - self.mean().powi(2)
            }
        }
    }
    fn fraction_below(&self, x: f64) -> f64 {
        match self {
            &Size::Exp(lambda) => 1.0 - f64::exp(-lambda * x),
            &Size::Pareto(alpha) => 1.0 - x.powf(-alpha),
            &Size::Hyper(low, high, low_prob) => {
                (1.0 - f64::exp((-1.0 / low) * x)) * low_prob
                    + (1.0 - f64::exp((-1.0 / high) * x)) * (1.0 - low_prob)
            }
        }
    }
    fn mean_given_below(&self, x: f64) -> f64 {
        match self {
            &Size::Exp(lambda) => (1.0 - f64::exp(-lambda * x) * (1.0 + x * lambda)) / lambda,
            &Size::Pareto(alpha) => (alpha / (alpha - 1.0)) * (1.0 - x.powf(1.0 - alpha)),
            &Size::Hyper(low, high, low_prob) => {
                Size::Exp(1.0 / low).mean_given_below(x) * low_prob
                    + Size::Exp(1.0 / high).mean_given_below(x) * (1.0 - low_prob)
            }
        }
    }
    fn descending_busy_period(&self, x: f64, l: f64) -> f64 {
        let integrand = &|t: f64|1.0/(1.0-l*self.mean_given_below(t));
        integrate(integrand, 0.0, x, 1e-6).integral
        //integrand(x)
    }
    fn mean_sq_below(&self, x: f64) -> f64 {
        match self {
            &Size::Exp(lambda) => {
                (2.0 + f64::exp(-x * lambda) * (-2.0 - x * lambda * (2.0 + x * lambda)))
                    / ((1.0 - f64::exp(-x * lambda)) * lambda * lambda)
            }
            _ => unimplemented!(),
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
    let size = Size::balanced_hyper(200.0);
    //let size = Size::Exp(1.0);
    //let size = Size::Pareto(2.01);
    println!(
        "Mean: {}, C^2: {}",
        size.mean(),
        size.variance() / size.mean().powf(2.0)
    );
    let rho = 0.9;
    let time = 1e5;
    let k = 10;

    let mut policies: Vec<Box<Dispatch>> = vec![Box::new(SITA::new(&size)), Box::new(LWL::new()), Box::new(Random::new(seed)), Box::new(JSQ::new()), Box::new(Cost::new())];
    println!(",{}", policies.iter().map(|p|format!("{}", p)).collect::<Vec<String>>().join(","));
    for rho in vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,/* 0.925, 0.95, 0.97, 0.98, 0.99*/] {
        let mut results = vec![rho];
        for policy in &mut policies {
        let lambda = rho / size.mean();
        let completions = simulate(time, lambda, &size, policy, k, seed);
        let mean = completions.iter().map(|c| c.response_time).sum::<f64>() / completions.len() as f64;
        results.push(mean);
        }
        println!("{}", results.iter().map(|p|format!("{}", p)).collect::<Vec<String>>().join(","));
    }
    /*
    print_sim_mean(time, rho, &size, &mut SITA::new(&size), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut Random::new(seed), k, seed);
    print_sim_mean(time, rho, &size, &mut JSQ::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut Cost::new(), k, seed);
    */
    /*
    print_sim_mean(time, rho, &size, &mut Random::new(seed), 1, seed);
    print_sim_mean(time, rho, &size, &mut JIQ::new(seed), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL_me::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut LWL_2me::new(), k, seed);
    print_sim_mean(time, rho, &size, &mut IMDbelow::new(8.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMDbelow::new(4.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMDbelow::new(2.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(8.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(4.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(2.0), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.5), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.2), k, seed);
    print_sim_mean(time, rho, &size, &mut IMD::new(1.1), k, seed);
    print_sim_mean(
        time,
        rho,
        &size,
        &mut Cost2::new(&size, rho / size.mean()),
        k,
        seed,
    );
    /*
    print_sim_mean(
        time,
        rho,
        &size,
        &mut Cost3::new(&size, rho / size.mean()),
        k,
        seed,
    );
    */
    */
}
