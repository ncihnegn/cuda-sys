#![warn(clippy::useless_attribute)]

use std::ffi::CStr;
use std::fmt;

use num::FromPrimitive;

use super::bind::*;

#[repr(u32)]
#[derive(Debug, FromPrimitive, PartialEq)]
pub enum ErrorCode {
    Success = 0,
    MissingConfiguration = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    LaunchFailure = 4,
    PriorLaunchFailure = 5,
    LaunchTimeout = 6,
    LaunchOutOfResources = 7,
    InvalidDeviceFunction = 8,
    InvalidConfiguration = 9,
    InvalidDevice = 10,
    InvalidValue = 11,
    InvalidPitchValue = 12,
    InvalidSymbol = 13,
    MapBufferObjectFailed = 14,
    UnmapBufferObjectFailed = 15,
    InvalidHostPointer = 16,
    InvalidDevicePointer = 17,
    InvalidTexture = 18,
    InvalidTextureBinding = 19,
    InvalidChannelDescriptor = 20,
    InvalidMemcpyDirection = 21,
    AddressOfConstant = 22,
    TextureFetchFailed = 23,
    TextureNotBound = 24,
    SynchronizationError = 25,
    InvalidFilterSetting = 26,
    InvalidNormSetting = 27,
    MixedDeviceExecution = 28,
    CudartUnloading = 29,
    Unknown = 30,
    NotYetImplemented = 31,
    MemoryValueTooLarge = 32,
    InvalidResourceHandle = 33,
    NotReady = 34,
    InsufficientDriver = 35,
    SetOnActiveProcess = 36,
    InvalidSurface = 37,
    NoDevice = 38,
    ECCUncorrectable = 39,
    SharedObjectSymbolNotFound = 40,
    SharedObjectInitFailed = 41,
    UnsupportedLimit = 42,
    DuplicateVariableName = 43,
    DuplicateTextureName = 44,
    DuplicateSurfaceName = 45,
    DevicesUnavailable = 46,
    InvalidKernelImage = 47,
    NoKernelImageForDevice = 48,
    IncompatibleDriverContext = 49,
    PeerAccessAlreadyEnabled = 50,
    PeerAccessNotEnabled = 51,
    DeviceAlreadyInUse = 54,
    ProfilerDisabled = 55,
    ProfilerNotInitialized = 56,
    ProfilerAlreadyStarted = 57,
    ProfilerAlreadyStopped = 58,
    Assert = 59,
    TooManyPeers = 60,
    HostMemoryAlreadyRegistered = 61,
    HostMemoryNotRegistered = 62,
    OperatingSystem = 63,
    PeerAccessUnsupported = 64,
    LaunchMaxDepthExceeded = 65,
    LaunchFileScopedTex = 66,
    LaunchFileScopedSurf = 67,
    SyncDepthExceeded = 68,
    LaunchPendingCountExceeded = 69,
    NotPermitted = 70,
    NotSupported = 71,
    HardwareStackError = 72,
    IllegalInstruction = 73,
    MisalignedAddress = 74,
    InvalidAddressSpace = 75,
    InvalidPc = 76,
    IllegalAddress = 77,
    InvalidPtx = 78,
    InvalidGraphicsContext = 79,
    NvlinkUncorrectable = 80,
    JitCompilerNotFound = 81,
    CooperativeLaunchTooLarge = 82,
    SystemNotReady = 83,
    IllegalState = 84,
    StartupFailure = 127,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalidated = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolation = 905,
    StreamCaptureImplicit = 906,
    CapturedEvent = 907,
    ApiFailureBase = 10000,
}

impl ErrorCode {
    pub fn check(c: u32) -> ErrorCode {
        match ErrorCode::from_u32(c) {
            Some(e) => e,
            None => panic!("Unknown error code: {}", c),
        }
    }
}

macro_rules! check_error {
    ($ec: expr, $ret: expr) => {
        match ErrorCode::check($ec) {
            ErrorCode::Success => Ok($ret),
            _ => Err(ErrorCode::check($ec)),
        }
    };
}

impl Default for cudaDeviceProp {
    fn default() -> cudaDeviceProp {
        cudaDeviceProp {
            name: [0; 256],
            uuid: CUuuid { bytes: [0; 16] },
            luid: [0; 8],
            luidDeviceNodeMask: 0,
            totalGlobalMem: 0,
            sharedMemPerBlock: 0,
            regsPerBlock: 0,
            warpSize: 0,
            memPitch: 0,
            maxThreadsPerBlock: 0,
            maxThreadsDim: [0; 3],
            maxGridSize: [0; 3],
            clockRate: 0,
            totalConstMem: 0,
            major: 0,
            minor: 0,
            textureAlignment: 0,
            texturePitchAlignment: 0,
            deviceOverlap: 0,
            multiProcessorCount: 0,
            kernelExecTimeoutEnabled: 0,
            integrated: 0,
            canMapHostMemory: 0,
            computeMode: 0,
            maxTexture1D: 0,
            maxTexture1DMipmap: 0,
            maxTexture1DLinear: 0,
            maxTexture2D: [0; 2],
            maxTexture2DMipmap: [0; 2],
            maxTexture2DLinear: [0; 3],
            maxTexture2DGather: [0; 2],
            maxTexture3D: [0; 3],
            maxTexture3DAlt: [0; 3],
            maxTextureCubemap: 0,
            maxTexture1DLayered: [0; 2],
            maxTexture2DLayered: [0; 3],
            maxTextureCubemapLayered: [0; 2],
            maxSurface1D: 0,
            maxSurface2D: [0; 2],
            maxSurface3D: [0; 3],
            maxSurface1DLayered: [0; 2],
            maxSurface2DLayered: [0; 3],
            maxSurfaceCubemap: 0,
            maxSurfaceCubemapLayered: [0; 2],
            surfaceAlignment: 0,
            concurrentKernels: 0,
            ECCEnabled: 0,
            pciBusID: 0,
            pciDeviceID: 0,
            pciDomainID: 0,
            tccDriver: 0,
            asyncEngineCount: 0,
            unifiedAddressing: 0,
            memoryClockRate: 0,
            memoryBusWidth: 0,
            l2CacheSize: 0,
            maxThreadsPerMultiProcessor: 0,
            streamPrioritiesSupported: 0,
            globalL1CacheSupported: 0,
            localL1CacheSupported: 0,
            sharedMemPerMultiprocessor: 0,
            regsPerMultiprocessor: 0,
            managedMemory: 0,
            isMultiGpuBoard: 0,
            multiGpuBoardGroupID: 0,
            hostNativeAtomicSupported: 0,
            singleToDoublePrecisionPerfRatio: 0,
            pageableMemoryAccess: 0,
            concurrentManagedAccess: 0,
            computePreemptionSupported: 0,
            canUseHostPointerForRegisteredMem: 0,
            cooperativeLaunch: 0,
            cooperativeMultiDeviceLaunch: 0,
            sharedMemPerBlockOptin: 0,
            pageableMemoryAccessUsesHostPageTables: 0,
            directManagedMemAccessFromHost: 0,
        }
    }
}

#[repr(i32)]
#[derive(Debug, FromPrimitive, PartialEq)]
pub enum ComputeMode {
    Default = 0,
    Exclusive = 1,
    Prohibited = 2,
    ExclusiveProcess = 3,
}

#[derive(Debug)]
pub struct DeviceProp {
    pub name: String,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: i32,
    pub warp_size: i32,
    pub mem_pitch: usize,
    pub max_threads_per_block: i32,
    pub max_threads_dim: [i32; 3],
    pub max_grid_size: [i32; 3],
    pub clock_rate: i32,
    pub total_const_mem: usize,
    pub version: Version,
    pub texture_alignment: usize,
    pub texture_pitch_alignment: usize,
    pub device_overlap: bool,
    pub multi_processor_count: i32,
    pub kernel_exec_timeout_enabled: bool,
    pub integrated: bool,
    pub can_map_host_memory: bool,
    pub compute_mode: ComputeMode,
    pub max_texture1d: i32,
    pub max_texture1d_mipmap: i32,
    pub max_texture1d_linear: i32,
    pub max_texture2d: [i32; 2],
    pub max_texture2d_mipmap: [i32; 2],
    pub max_texture2d_linear: [i32; 3],
    pub max_texture2d_gather: [i32; 2],
    pub max_texture3d: [i32; 3],
    pub max_texture3d_alt: [i32; 3],
    pub max_texture_cubemap: i32,
    pub max_texture1d_layered: [i32; 2],
    pub max_texture2d_layered: [i32; 3],
    pub max_texture_cubemap_layered: [i32; 2],
    pub max_surface1d: i32,
    pub max_surface2d: [i32; 2],
    pub max_surface3d: [i32; 3],
    pub max_surface1d_layered: [i32; 2],
    pub max_surface2d_layered: [i32; 3],
    pub max_surface_cubemap: i32,
    pub max_surface_cubemap_layered: [i32; 2],
    pub surface_alignment: usize,
    pub concurrent_kernels: bool,
    pub ecc_enabled: bool,
    pub pci_bus_id: i32,
    pub pci_device_id: i32,
    pub pci_domain_id: i32,
    pub tcc_driver: bool,
    pub async_engine_count: i32,
    pub unified_addressing: bool,
    pub memory_clock_rate: i32,
    pub memory_bus_width: i32,
    pub l2_cache_size: i32,
    pub max_threads_per_multi_processor: i32,
    pub stream_priorities_supported: bool,
    pub global_l1_cache_supported: bool,
    pub local_l1_cache_supported: bool,
    pub shared_mem_per_multiprocessor: usize,
    pub regs_per_multiprocessor: i32,
    pub managed_memory: bool,
    pub is_multi_gpu_board: bool,
    pub multi_gpu_board_group_id: i32,
    pub host_native_atomic_supported: bool,
    pub single_to_double_precision_perf_ratio: i32,
    pub pageable_memory_access: bool,
    pub concurrent_managed_access: bool,
    pub compute_preemption_supported: bool,
    pub can_use_host_pointer_for_registered_mem: bool,
    pub cooperative_launch: bool,
    pub cooperative_multidevice_launch: bool,
    pub shared_mem_per_block_optin: usize,
    pub pageable_memory_access_uses_host_page_tables: bool,
    pub direct_managed_mem_access_from_host: bool,
}

impl DeviceProp {
    pub fn from(prop: &cudaDeviceProp) -> Self {
        DeviceProp {
            name: unsafe { CStr::from_ptr(&prop.name[0] as *const i8) }
                .to_str()
                .unwrap_or("")
                .to_owned(),
            total_global_mem: prop.totalGlobalMem,
            shared_mem_per_block: prop.sharedMemPerBlock,
            regs_per_block: prop.regsPerBlock,
            warp_size: prop.warpSize,
            mem_pitch: prop.memPitch,
            max_threads_per_block: prop.maxThreadsPerBlock,
            max_threads_dim: prop.maxThreadsDim,
            max_grid_size: prop.maxGridSize,
            clock_rate: prop.clockRate,
            total_const_mem: prop.totalConstMem,
            version: Version {
                major: prop.major,
                minor: prop.minor,
            },
            texture_alignment: prop.textureAlignment,
            texture_pitch_alignment: prop.texturePitchAlignment,
            device_overlap: prop.deviceOverlap == 1,
            multi_processor_count: prop.multiProcessorCount,
            kernel_exec_timeout_enabled: prop.kernelExecTimeoutEnabled == 1,
            integrated: prop.integrated == 1,
            can_map_host_memory: prop.canMapHostMemory == 1,
            compute_mode: ComputeMode::from_i32(prop.computeMode)
                .unwrap_or(ComputeMode::Prohibited),
            max_texture1d: prop.maxTexture1D,
            max_texture1d_mipmap: prop.maxTexture1DMipmap,
            max_texture1d_linear: prop.maxTexture1DLinear,
            max_texture2d: prop.maxTexture2D,
            max_texture2d_mipmap: prop.maxTexture2DMipmap,
            max_texture2d_linear: prop.maxTexture2DLinear,
            max_texture2d_gather: prop.maxTexture2DGather,
            max_texture3d: prop.maxTexture3D,
            max_texture3d_alt: prop.maxTexture3DAlt,
            max_texture_cubemap: prop.maxTextureCubemap,
            max_texture1d_layered: prop.maxTexture1DLayered,
            max_texture2d_layered: prop.maxTexture2DLayered,
            max_texture_cubemap_layered: prop.maxTextureCubemapLayered,
            max_surface1d: prop.maxSurface1D,
            max_surface2d: prop.maxSurface2D,
            max_surface3d: prop.maxSurface3D,
            max_surface1d_layered: prop.maxSurface1DLayered,
            max_surface2d_layered: prop.maxSurface2DLayered,
            max_surface_cubemap: prop.maxSurfaceCubemap,
            max_surface_cubemap_layered: prop.maxSurfaceCubemapLayered,
            surface_alignment: prop.surfaceAlignment,
            concurrent_kernels: prop.concurrentKernels == 1,
            ecc_enabled: prop.ECCEnabled == 1,
            pci_bus_id: prop.pciBusID,
            pci_device_id: prop.pciDeviceID,
            pci_domain_id: prop.pciDomainID,
            tcc_driver: prop.tccDriver == 1,
            async_engine_count: prop.asyncEngineCount,
            unified_addressing: prop.unifiedAddressing == 1,
            memory_clock_rate: prop.memoryClockRate,
            memory_bus_width: prop.memoryBusWidth,
            l2_cache_size: prop.l2CacheSize,
            max_threads_per_multi_processor: prop.maxThreadsPerMultiProcessor,
            stream_priorities_supported: prop.streamPrioritiesSupported == 1,
            global_l1_cache_supported: prop.globalL1CacheSupported == 1,
            local_l1_cache_supported: prop.localL1CacheSupported == 1,
            shared_mem_per_multiprocessor: prop.sharedMemPerMultiprocessor,
            regs_per_multiprocessor: prop.regsPerMultiprocessor,
            managed_memory: prop.managedMemory == 1,
            is_multi_gpu_board: prop.isMultiGpuBoard == 1,
            multi_gpu_board_group_id: prop.multiGpuBoardGroupID,
            host_native_atomic_supported: prop.hostNativeAtomicSupported == 1,
            single_to_double_precision_perf_ratio: prop.singleToDoublePrecisionPerfRatio,
            pageable_memory_access: prop.pageableMemoryAccess == 1,
            concurrent_managed_access: prop.concurrentManagedAccess == 1,
            compute_preemption_supported: prop.computePreemptionSupported == 1,
            can_use_host_pointer_for_registered_mem: prop.canUseHostPointerForRegisteredMem == 1,
            cooperative_launch: prop.cooperativeLaunch == 1,
            cooperative_multidevice_launch: prop.cooperativeMultiDeviceLaunch == 1,
            shared_mem_per_block_optin: prop.sharedMemPerBlockOptin,
            pageable_memory_access_uses_host_page_tables: prop
                .pageableMemoryAccessUsesHostPageTables
                == 1,
            direct_managed_mem_access_from_host: prop.directManagedMemAccessFromHost == 1,
        }
    }
}

pub struct Version {
    pub major: i32,
    pub minor: i32,
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}.{:?}", self.major, self.minor)
    }
}

impl Version {
    pub fn from_i32(v: i32) -> Version {
        Version {
            major: v / 1000,
            minor: (v % 1000) / 10,
        }
    }
}

#[repr(u32)]
#[derive(ToPrimitive)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

#[cfg(test)]
mod tests {
    use num::ToPrimitive;

    use super::*;

    #[test]
    fn error_code() {
        assert_eq!(ErrorCode::from_u32(0), Some(ErrorCode::Success));
    }

    #[test]
    fn memcpy_kind() {
        assert_eq!(MemcpyKind::Default.to_u32(), Some(4));
    }
}
