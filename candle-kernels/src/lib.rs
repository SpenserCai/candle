// Conditional module inclusion based on build configuration
#[cfg(not(candle_cuda_cubin))]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[cfg(candle_cuda_cubin)]
mod cubin {
    include!(concat!(env!("OUT_DIR"), "/cubin.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

/// CUDA module data format
/// Represents either PTX (text) or CUBIN (binary) kernel data
#[derive(Debug, Clone, Copy)]
pub enum ModuleData {
    /// PTX format - intermediate representation
    /// Requires JIT compilation at runtime, but is architecture-independent
    Ptx(&'static str),
    
    /// CUBIN format - pre-compiled binary
    /// No JIT compilation needed, but is architecture-specific
    Cubin(&'static [u8]),
}

/// A CUDA kernel module that can be loaded at runtime
pub struct Module {
    index: usize,
    data: ModuleData,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the module data format
    pub fn data(&self) -> &ModuleData {
        &self.data
    }

    /// Get module data as bytes (works for both PTX and CUBIN)
    /// This is the recommended method for loading modules
    pub fn as_bytes(&self) -> &'static [u8] {
        match &self.data {
            ModuleData::Ptx(s) => s.as_bytes(),
            ModuleData::Cubin(b) => b,
        }
    }

    /// Returns PTX string for PTX modules
    /// For CUBIN modules, this will panic with a helpful error message
    pub fn ptx(&self) -> &'static str {
        match &self.data {
            ModuleData::Ptx(s) => s,
            ModuleData::Cubin(_) => panic!(
                "Module contains CUBIN data, not PTX.\n\
                 Use Module::as_bytes() instead for compatibility with both formats."
            ),
        }
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

// Conditional macro definition based on build configuration
#[cfg(not(candle_cuda_cubin))]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            data: ModuleData::Ptx(ptx::$cst),
        };
    };
}

#[cfg(candle_cuda_cubin)]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            data: ModuleData::Cubin(cubin::$cst),
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);
