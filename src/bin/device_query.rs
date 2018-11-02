extern crate cuda_sys;

use cuda_sys::runtime::*;

fn main() {
    let driver_version = driver_get_version();
    let runtime_version = runtime_get_version();
    println!(
        "Driver version: {:?}, Runtime version: {:?}",
        driver_version.unwrap(),
        runtime_version.unwrap()
    );
    let device_count = get_device_count().unwrap_or(0);
    println!("Device count: {}", device_count);
    for i in 0..device_count {
        let prop = get_device_prop(i);
        if i > 0 {
            println!();
        }
        println!("Device {}: {:?}", i, prop);
    }
}
