#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod bindings;

use std::ffi::CString;
use std::io::Write;
use std::ptr::{addr_of_mut, null, null_mut};
use std::sync::{Arc, Mutex};

use crate::bindings::{feature_node, free_and_destroy_model, load_model, save_model};
use eyre::{eyre, Report, WrapErr};

#[derive(Debug)]
pub struct Model {
    model: *mut bindings::model,
    bias: f64,
    num_features: i32,

    owned: bool,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load_from_file(path: &str) -> Result<Self, Report> {
        let c_path = CString::new(path).wrap_err("invalid c-string")?;

        let model = unsafe { load_model(c_path.as_ptr()) };

        unsafe { Self::new(model) }
    }

    pub fn load_from_str(def: &str) -> Result<Self, Report> {
        let mut temp = tempfile::NamedTempFile::new().wrap_err("error creating tempfile")?;
        temp.write_all(def.as_bytes())
            .wrap_err("error writing to tempfile")?;

        Model::load_from_file(temp.path().to_str().unwrap())
    }

    pub unsafe fn new(model: *mut bindings::model) -> Result<Self, Report> {
        if model.is_null() {
            return Err(eyre!("null model pointer"));
        }

        Ok(Self {
            model,
            bias: (*model).bias,
            num_features: (*model).nr_feature,
            owned: false,
        })
    }

    pub(crate) unsafe fn new_owned(model: *mut bindings::model) -> Self {
        Self {
            model,
            bias: (*model).bias,
            num_features: (*model).nr_feature,
            owned: true,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Report> {
        let c_path = CString::new(path).wrap_err("invalid c-string")?;

        let res = unsafe { save_model(c_path.as_ptr(), self.model) };

        if res == 0 {
            Ok(())
        } else {
            Err(eyre!("error saving model, code: {res}"))
        }
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    pub fn num_features(&self) -> i32 {
        self.num_features
    }

    pub fn predict(&self, features: &[feature_node]) -> f64 {
        unsafe { bindings::predict(self.model, features.as_ptr()) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.owned {
            unsafe { free_and_destroy_model(&mut self.model) }
        }
    }
}

pub struct OwnedModel {
    owned_model: bindings::model,
    model: Option<Model>,
    labels: Vec<i32>,
    w: Vec<f64>,
}

impl OwnedModel {
    pub fn new(owned_model: bindings::model, labels: Vec<i32>, w: Vec<f64>) -> Self {
        let mut model = OwnedModel {
            owned_model,
            labels,
            w,
            model: None,
        };

        model.owned_model.w = model.w.as_mut_ptr();
        model.owned_model.label = model.labels.as_mut_ptr();
        model.model = unsafe {
            Some(Model::new_owned(
                &mut model.owned_model as *mut bindings::model,
            ))
        };

        model
    }

    pub fn model(&self) -> &Model {
        self.model.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::bindings::LIBLINEAR_VERSION;

    #[test]
    fn version() {
        assert_eq!(LIBLINEAR_VERSION, 245)
    }
}
