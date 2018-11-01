use super::runtime::data_type::*;
use super::runtime::*;

impl ComputeCapability {
    fn num_cores(&self) -> i32 {
        match self.major {
            3 => 192,
            5 => 128,
            6 => {
                if self.minor == 0 {
                    64
                } else {
                    128
                }
            }
            7 => 64,
            _ => panic!("Unknown ComputeCapability {:?}", self),
        }
    }
}

impl DeviceProp {
    fn performance(&self) -> i64 {
        i64::from(self.version.num_cores() * self.multi_processor_count)
            * i64::from(self.clock_rate)
    }
}

pub fn performance(id: i32) -> i64 {
    get_device_prop(id).map(|p| p.performance()).unwrap_or(0)
}

pub fn fastest_device_id() -> Option<i32> {
    let device_count = get_device_count().unwrap_or(0);
    println!("Device count: {}", device_count);
    (0..device_count)
        .map(get_device_prop)
        .enumerate()
        .filter(|&(_, ref r)| r.is_ok())
        .map(|(i, r)| (i, r.unwrap()))
        .filter(|&(_, ref p)| p.compute_mode != ComputeMode::Prohibited)
        .map(|(i, p)| (i, p.performance()))
        .max_by_key(|&(_, perf)| perf)
        .map(|(i, _)| i as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_best_gpu() {
        let fastest = fastest_device_id();
        let device_count = get_device_count().unwrap_or(0);
        fastest.map(|x| {
            for i in 0..device_count {
                assert!(performance(i) <= performance(x));
            }
        });
    }
}
