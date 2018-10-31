use std::os::raw::c_int;

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
mod bind;

mod data_type;

use self::bind::*;
use self::data_type::*;

pub fn get_device_count() -> i32 {
    let mut device_count: c_int = 0;
    unsafe {
        cudaGetDeviceCount(&mut device_count);
    }
    return device_count;
}

pub fn set_device(id: i32) {
    unsafe {
        cudaSetDevice(id);
    }
}

pub fn get_device_prop(id: i32) -> DeviceProp {
    let mut prop = cudaDeviceProp::default();
    unsafe {
        cudaGetDeviceProperties(&mut prop, id);
    }
    DeviceProp::from(&prop)
}
