extern crate cuda_sys;

use cuda_sys::runtime::*;

fn main() {
    let device_count = get_device_count();
    println!("Device count: {}", device_count);
    for i in 0..device_count {
        let prop = get_device_prop(i);
        println!("Device {}: {:?}", i, prop);
    }
}
