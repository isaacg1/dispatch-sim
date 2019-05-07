#![allow(warnings)]
extern crate rand;
use rand::distributions::Exp;
use rand::prelude::*;
use rand::prng::isaac::IsaacRng;

extern crate quadrature;
use quadrature::integrate;

extern crate noisy_float;
use noisy_float::prelude::*;

use std::fmt;

use std::f64::INFINITY;

use std::collections::HashMap;

const EPSILON: f64 = 1e-10;
const toggle: bool = false;
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
    fn dispatch(&mut self, job_size: f64, queues: &Vec<Vec<Job>>, candidates: &Vec<usize>)
        -> usize;
}

impl<S: Dispatch + ?Sized> Dispatch for Box<S> {
    fn dispatch(
        &mut self,
        job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        (**self).dispatch(job_size, queues, candidates)
    }
}

#[derive(Clone, Debug)]
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
    fn dispatch(
        &mut self,
        _job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        *candidates.choose(&mut self.rng).unwrap()
    }
}

impl fmt::Display for Random {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Random")
    }
}

#[derive(Clone)]
struct JSQ {}

impl JSQ {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for JSQ {
    fn dispatch(
        &mut self,
        _job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        *candidates.iter().min_by_key(|&&c| queues[c].len()).unwrap()
    }
}

impl fmt::Display for JSQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JSQ")
    }
}

#[derive(Clone)]
struct JSQ_d {
    rng: IsaacRng,
    d: usize,
}

impl JSQ_d {
    fn new(seed: u64, d: usize) -> Self {
        Self {
            rng: IsaacRng::new_from_u64(seed),
            d,
        }
    }
}

impl Dispatch for JSQ_d {
    fn dispatch(
        &mut self,
        _job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        let observed_candidates: Vec<usize> = candidates
            .choose_multiple(&mut self.rng, self.d)
            .cloned()
            .collect();
        observed_candidates
            .into_iter()
            .min_by_key(|&c| queues[c].len())
            .unwrap()
    }
}

impl fmt::Display for JSQ_d {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JSQ_d({})", self.d)
    }
}

#[derive(Clone, Debug)]
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
    fn dispatch(
        &mut self,
        _job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        candidates
            .iter()
            .cloned()
            .filter(|&c| queues[c].is_empty())
            .choose(&mut self.rng)
            .unwrap_or_else(|| *candidates.choose(&mut self.rng).unwrap())
    }
}

impl fmt::Display for JIQ {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JIQ")
    }
}

#[derive(Clone)]
struct LWL {}

impl LWL {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for LWL {
    fn dispatch(
        &mut self,
        _job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        candidates
            .iter()
            .map(|&c| (c, queues[c].iter().map(|j| j.rem_size).sum::<f64>()))
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

struct RR {
    dispatches: Vec<usize>,
}

impl RR {
    fn new(k: usize) -> Self {
        Self {
            dispatches: (0..k).collect(),
        }
    }
}

impl Dispatch for RR {
    fn dispatch(
        &mut self,
        job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        let (i, &dispatch) = self
            .dispatches
            .iter()
            .enumerate()
            .find(|(_, d)| candidates.contains(d))
            .unwrap();
        if toggle {
            println!(
                "{:?} {:?} i={} {} {}",
                self.dispatches, candidates, i, dispatch, job_size
            );
        }
        let dispatch_verify = self.dispatches.remove(i);
        assert_eq!(dispatch, dispatch_verify);
        self.dispatches.push(dispatch);
        return dispatch;
    }
}

impl fmt::Display for RR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RR")
    }
}

struct IMD {}
impl IMD {
    fn new() -> Self {
        Self {}
    }
}

impl Dispatch for IMD {
    fn dispatch(&mut self, _: f64, _: &Vec<Vec<Job>>, _: &Vec<usize>) -> usize {
        0
    }
}

impl fmt::Display for IMD {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IMD")
    }
}

struct SITA {
    thresholds: Vec<f64>,
}

impl SITA {
    fn new(thresholds: Vec<f64>) -> Self {
        Self { thresholds }
    }
}
impl Dispatch for SITA {
    fn dispatch(
        &mut self,
        job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        let preferred = self
            .thresholds
            .iter()
            .position(|&t| t > job_size)
            .unwrap_or(queues.len());
        let mut mut_candidates = candidates.clone();
        mut_candidates.sort_by_key(|&c| (c as isize - preferred as isize).abs());
        mut_candidates[0]
    }
}
impl fmt::Display for SITA {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SITA")
    }
}

struct Split {
    rng: IsaacRng,
    split_point: f64,
    low_weight: f64,
}

impl Split {
    fn new(seed: u64, split_point: f64, low_weight: f64) -> Self {
        Self {
            rng: IsaacRng::new_from_u64(seed),
            split_point,
            low_weight,
        }
    }
}

impl Dispatch for Split {
    fn dispatch(
        &mut self,
        job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        let float_preferred = if job_size < self.split_point {
            self.rng.gen_range(0., self.low_weight)
        } else {
            self.rng.gen_range(self.low_weight, 1.0)
        };
        let preferred = (float_preferred * queues.len() as f64).floor() as usize;
        let mut mut_candidates = candidates.clone();
        mut_candidates.sort_by_key(|&c| (c as isize - preferred as isize).abs());
        mut_candidates[0]
    }
}
impl fmt::Display for Split {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Split")
    }
}

struct FPI {
    size_dist: Size,
    lambda: f64,
}

impl FPI {
    fn new(size_dist: Size, rho: f64) -> Self {
        Self {
            size_dist,
            lambda: rho / size_dist.mean(),
        }
    }
    fn custom_integral(&self, size: f64) -> f64 {
        match self.size_dist {
            Size::BoundedPareto(1.5, bound) => {
                /* (1/(9 (1 - 1/b^(3/2)) b^(
                 *  9/2) lambda^2))(-1 + b^(
                 *    3/2))^2 (-((3 b^(3/2) lambda (1 + b^(3/2) (-1 + 3 lambda)))/(
                 *     b^(3/2) (3 lambda (-1 + Sqrt[t]) - Sqrt[t]) + Sqrt[t])) - (
                 *    3 b^(3/2) lambda)/Sqrt[t] -
                 *    2 (1 + b^(3/2) (-1 + 3 lambda)) Log[
                 *      b^(3/2) (3 lambda (-1 + Sqrt[t]) - Sqrt[t]) + Sqrt[t]] + (1 +
                 *       b^(3/2) (-1 + 3 lambda)) Log[t])
                 */
                if size > bound {
                    return self.custom_integral(bound);
                }
                let alpha = 1.5;
                let num1 = (bound.powf(alpha) - 1.0).powi(2);
                let bound_alpha = bound * bound.sqrt();
                let root_size = size.sqrt();

                let denom1 =
                    9.0 * (1.0 - 1.0 / bound_alpha) * bound_alpha.powi(3) * self.lambda.powi(2);
                let subterm = 1.0 + bound_alpha * (3.0 * self.lambda - 1.0);
                let term1_num = 3.0 * bound_alpha * self.lambda * subterm;
                let term1_denom =
                    bound_alpha * (3.0 * self.lambda * (root_size - 1.0) - root_size) + root_size;
                let term2 = 3.0 * bound_alpha * self.lambda / root_size;
                let term3 = 2.0 * subterm * (-term1_denom).ln();
                let term4 = subterm * size.ln();
                (num1 / denom1) * (-(term1_num / term1_denom) - term2 - term3 + term4)
            }
            _ => unimplemented!(),
        }
    }
}

impl Dispatch for FPI {
    fn dispatch(
        &mut self,
        job_size: f64,
        queues: &Vec<Vec<Job>>,
        candidates: &Vec<usize>,
    ) -> usize {
        let alternative = *candidates
            .iter()
            .min_by_key(|i| {
                n64({
                    let mut queue = queues[**i].clone();
                    queue.sort_by_key(|j| n64(j.rem_size));
                    let mut delta = 0.0;
                    let mut previous_threshold = job_size;
                    let mut work_below = 0.0;
                    for old_job in queue {
                        if old_job.rem_size <= job_size {
                            delta += old_job.rem_size
                                / (1.0 - self.lambda * self.size_dist.mean_given_below(job_size));
                        } else {
                            delta += job_size
                                / (1.0
                                    - self.lambda
                                        * self.size_dist.mean_given_below(old_job.rem_size));
                        }
                        if old_job.rem_size >= job_size {
                            assert!(old_job.rem_size >= previous_threshold);
                            let custom_delta = self.custom_integral(old_job.rem_size)
                                - self.custom_integral(previous_threshold);
                            delta += custom_delta * (work_below + job_size).powi(2) / 2.0
                                - custom_delta * work_below.powi(2) / 2.0;
                            previous_threshold = old_job.rem_size;
                        }
                        work_below += old_job.rem_size;
                    }
                    let custom_delta =
                        self.custom_integral(INFINITY) - self.custom_integral(previous_threshold);
                    delta += custom_delta * (work_below + job_size).powi(2) / 2.0
                        - custom_delta * work_below.powi(2) / 2.0;
                    delta
                })
            })
            .unwrap();
        let primary = *candidates
            .iter()
            .min_by_key(|i| {
                n64({
                    let mut queue = queues[**i].clone();
                    queue.sort_by_key(|j| n64(j.rem_size));
                    let mut delta = 0.0;
                    let mut previous_threshold = job_size;
                    let mut work_below = 0.0;
                    let mut addends = 0.0;
                    for old_job in queue {
                        if old_job.rem_size <= job_size {
                            delta += old_job.rem_size
                                / (1.0 - self.lambda * self.size_dist.mean_given_below(job_size));
                        } else {
                            delta += job_size
                                / (1.0
                                    - self.lambda
                                        * self.size_dist.mean_given_below(old_job.rem_size));
                        }
                        if old_job.rem_size >= job_size {
                            assert!(old_job.rem_size >= previous_threshold);
                            let custom_delta = self.custom_integral(old_job.rem_size)
                                - self.custom_integral(previous_threshold);
                            delta += custom_delta * work_below * job_size;
                            addends += custom_delta * job_size.powi(2) / 2.0;
                            previous_threshold = old_job.rem_size;
                        }
                        work_below += old_job.rem_size;
                    }
                    let custom_delta =
                        self.custom_integral(INFINITY) - self.custom_integral(previous_threshold);
                    delta += custom_delta * work_below * job_size;
                    delta
                })
            })
            .unwrap();
        assert!(primary == alternative);
        primary
    }
}

impl fmt::Display for FPI {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FPI")
    }
}

fn simulate(
    end_time: f64,
    rho: f64,
    size_dist: &Size,
    dispatcher: &mut impl Dispatch,
    k: usize,
    g: Option<f64>,
    seed: u64,
) -> Vec<Completion> {
    let lambda = rho / size_dist.mean();
    let guard_c: f64 = 1.0 + 1.0 / (1.0 + (1.0 / (1.0 - rho)).ln());
    //let guard_c: f64 = 1.0 + 1.0 / (0.5 + (1.0 / (1.0 - rho)).log(100.0));
    //let guard_c = 2.0;
    let guardrail_multiplier = g;

    let mut current_time: f64 = 0.;
    let mut queues: Vec<Vec<Job>> = vec![vec![]; k];
    let mut set_to_empty: Vec<bool> = vec![false; k];
    let mut completions: Vec<Completion> = vec![];

    let arrival_generator = Exp::new(lambda);
    let mut rng = IsaacRng::new_from_u64(seed);
    let mut rng2 = IsaacRng::new_from_u64(seed);
    let mut arrival_increment = arrival_generator.sample(&mut rng);

    let mut work_in_ranks: HashMap<i32, Vec<f64>> = HashMap::new();
    let mut num_bad_dispatches = 0;

    while current_time < end_time
        || queues
            .iter()
            .any(|q| q.iter().any(|j| j.arrival_time < end_time))
    {
        queues
            .iter_mut()
            .for_each(|q| q.sort_by_key(|j| n64(j.rem_size)));
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
        if let Some(guardrail_multiplier_f) = guardrail_multiplier {
            for (i, queue) in queues.iter().enumerate() {
                if !set_to_empty[i] && queue.is_empty() {
                    if toggle {
                        if dispatcher.to_string() == "RR".to_string() {
                            println!("{} reset", i);
                        }
                    }
                    for work_in_rank in work_in_ranks.values_mut() {
                        let min = *work_in_rank
                            .iter()
                            .min_by(|a, b| a.partial_cmp(b).unwrap())
                            .unwrap();
                        work_in_rank[i] = min;
                    }
                    set_to_empty[i] = true;
                }
            }
        }
        if arrival_occured {
            let new_size = size_dist.sample(&mut rng);
            let candidates: Vec<usize> = if let Some(guardrail_multiplier_f) = guardrail_multiplier
            {
                let rank = new_size.log(guard_c).floor() as i32;
                let work_in_rank = work_in_ranks.entry(rank).or_insert_with(|| vec![0.0; k]);
                if toggle {
                    if dispatcher.to_string() == "RR".to_string() {
                        println!(
                            "{:?} {} {}",
                            work_in_rank,
                            rank,
                            guardrail_multiplier_f * guard_c.powi(rank + 1)
                        );
                    }
                }
                let min = work_in_rank
                    .iter()
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                (0..queues.len())
                    .filter(|&j| {
                        work_in_rank[j] - min + new_size
                            < guardrail_multiplier_f * guard_c.powi(rank + 1)
                    })
                    .collect()
            } else {
                (0..queues.len()).collect()
            };
            assert!(!candidates.is_empty());
            let i = if dispatcher.to_string() != "IMD".to_string() {
                dispatcher.dispatch(new_size, &queues, &candidates)
            } else {
                let rank = new_size.log(guard_c).floor() as i32;
                let work_in_rank = &work_in_ranks[&rank];
                let min = work_in_rank
                    .iter()
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                let i_mins = work_in_rank
                    .iter()
                    .enumerate()
                    .filter(|&(i, w)| w == min)
                    .map(|(i, w)| i);
                let i_min = i_mins.choose(&mut rng2).unwrap();
                i_min
            };
            arrival_increment = arrival_generator.sample(&mut rng);

            queues[i].push(Job::new(new_size, current_time));
            if guardrail_multiplier.is_some() {
                let rank = new_size.log(guard_c).floor() as i32;
                set_to_empty[i] = false;
                work_in_ranks.get_mut(&rank).unwrap()[i] += new_size;
            }
        }
    }
    //Treat all jobs unfinished at end as immediately completing
    for queue in queues {
        for job in queue {
            completions.push(Completion::from_job(job, current_time));
        }
    }
    /*
    println!(
        "bad: {}, c: {}, g: {:?}, disp: {}",
        num_bad_dispatches as f64 / completions.len() as f64,
        guard_c,
        guardrail_multiplier,
        dispatcher
    );
    */
    completions
}
#[derive(Copy, Clone, Debug)]
enum Size {
    Exp(f64),
    Pareto(f64),
    BoundedPareto(f64, f64),
    Hyper(f64, f64, f64),
    Bimodal(f64, f64, f64),
    Trimodal(f64, f64, f64, f64, f64),
}

// Create bimodal
impl Distribution<f64> for Size {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            &Size::Exp(lambda) => {
                let dist = Exp::new(lambda);
                dist.sample(rng)
            }
            &Size::Pareto(alpha) => rng.gen_range(0., 1.).powf(-1. / alpha),
            &Size::BoundedPareto(alpha, bound) => loop {
                let out = rng.gen_range(0., 1.).powf(-1. / alpha);
                if out <= bound {
                    return out;
                }
            },
            &Size::Hyper(low, high, low_prob) => {
                let mean = if rng.gen_range(0., 1.) < low_prob {
                    low
                } else {
                    high
                };
                let dist = Exp::new(1.0 / mean);
                dist.sample(rng)
            }
            &Size::Bimodal(low, high, low_prob) => {
                if rng.gen_range(0., 1.) < low_prob {
                    low
                } else {
                    high
                }
            }
            &Size::Trimodal(low, med, high, low_prob, low_or_med_prob) => {
                let p = rng.gen_range(0., 1.);
                if p < low_prob {
                    low
                } else if p < low_or_med_prob {
                    med
                } else {
                    high
                }
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
            &Size::Bimodal(low, high, low_prob) => low * low_prob + high * (1.0 - low_prob),
            &Size::Trimodal(low, med, high, low_prob, low_or_med_prob) => {
                low * low_prob + med * (low_or_med_prob - low_prob) + high * (1.0 - low_or_med_prob)
            }
            &Size::BoundedPareto(alpha, bound) => {
                1.0 / (1.0 - bound.powf(-alpha)) * alpha / (alpha - 1.0)
                    * (1.0 - bound.powf(1.0 - alpha))
            }
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
            &Size::Bimodal(low, high, low_prob) => {
                low * low * low_prob + high * high * (1.0 - low_prob) - self.mean().powi(2)
            }
            &Size::Trimodal(low, med, high, low_prob, low_or_med_prob) => {
                low * low * low_prob
                    + med * med * (low_or_med_prob - low_prob)
                    + high * high * (1.0 - low_or_med_prob)
                    - self.mean().powi(2)
            }
            &Size::BoundedPareto(alpha, bound) => {
                1.0 / (1.0 - bound.powf(-alpha)) * alpha / (alpha - 2.0)
                    * (1.0 - bound.powf(2.0 - alpha))
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
            &Size::Bimodal(low, high, low_prob) => unimplemented!(),
            &Size::Trimodal(low, med, high, low_prob, low_or_med_prob) => unimplemented!(),
            &Size::BoundedPareto(_, _) => unimplemented!(),
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
            &Size::Bimodal(low, high, low_prob) => unimplemented!(),
            &Size::Trimodal(low, med, high, low_prob, low_or_med_prob) => unimplemented!(),
            &Size::BoundedPareto(alpha, bound) => {
                assert!(x <= bound);
                (alpha / (alpha - 1.0)) * (1.0 - x.powf(1.0 - alpha)) / (1.0 - bound.powf(-alpha))
            }
        }
    }
    fn descending_busy_period(&self, x: f64, l: f64) -> f64 {
        let integrand = &|t: f64| 1.0 / (1.0 - l * self.mean_given_below(t));
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
    g: Option<f64>,
    seed: u64,
) {
    let lambda = rho / size.mean();
    let completions = simulate(end_time, lambda, size, dispatcher, k, g, seed);
    let mean = completions.iter().map(|c| c.response_time).sum::<f64>() / completions.len() as f64;
    println!(
        "{:?}, {}, {}, {}, {}: {}",
        size, rho, k, end_time, dispatcher, mean
    );
}
fn main() {
    let time = 1e6;
    let k = 10;
    //let g = None;
    //let g = Some(2.0);

    println!("time={}", time);
    for seed in 0..10 {
        for size in vec![
            Size::Bimodal(1.0, 1000.0, 0.9995),
            Size::BoundedPareto(1.5, 10.0.powi(6)),
        ] {
            println!("{:?} {}", size, seed);
            println!(
                "g: {:?}, k: {}, Mean: {}, C^2: {}",
                "Set further on",
                k,
                size.mean(),
                size.variance() / size.mean().powf(2.0)
            );
            let mut to_print = true;
            let standard_rhos = vec![
                0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97,
                0.98, 0.99, 0.995, 0.9975, 0.999,
            ];
            let small_rhos = vec![
                0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26,
            ];
            for g in vec![None, Some(1.0)] {
                println!("g={:?}", g);
                for rho in vec![0.5] {
                    let mut results = vec![rho];
                    let mut policies: Vec<Box<Dispatch>> = vec![
                        Box::new(LWL::new()),
                        Box::new(Random::new(seed)),
                        Box::new(JSQ::new()),
                        Box::new(RR::new(k)),
                        Box::new(JSQ_d::new(seed, 2)),
                    ];
                    if let Size::BoundedPareto(_, _) = size {
                        policies.push(
                            Box::new(SITA::new(vec![
                                1.2343, 1.5617, 2.0391, 2.7741, 3.9920, 6.2313, 11.059, 24.801,
                                98.224,
                            ])));
                        policies.push(Box::new(FPI::new(size, rho)));
                    }
                    if let Size::Bimodal(_, _, _) = size {
                            policies.push(Box::new(Split::new(seed, 10.0, 0.9995 / 1.4995)));
                    }
                    if to_print {
                        println!(
                            ",{}",
                            policies
                                .iter()
                                .map(|p| format!("{}", p))
                                .collect::<Vec<String>>()
                                .join(",")
                        );
                        to_print = false;
                    }

                    for (i, policy) in policies.iter_mut().enumerate() {
                        let completions = simulate(time, rho, &size, policy, k, g, seed);
                        let mean = completions.iter().map(|c| c.response_time).sum::<f64>()
                            / completions.len() as f64;
                        results.push(mean);
                    }
                    println!(
                        "{}",
                        results
                            .iter()
                            .map(|p| format!("{:.6}", p))
                            .collect::<Vec<String>>()
                            .join(","),
                    );
                }
            }
        }
    }
}
