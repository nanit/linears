use std::ptr::null_mut;

use crate::{bindings, Model};

pub struct OwnedModel {
    owned_model: bindings::model,
    model: Option<Model>,
    labels: Vec<i32>,
    w: Vec<f64>,
}

impl OwnedModel {
    pub fn new(solver_type: i32, num_classes: i32, num_features: i32, bias: f64, labels: Vec<i32>, w: Vec<f64>) -> Self {
        
        let p = bindings::parameter {
            solver_type,

            eps: 0.0,
            C: 0.0,
            nr_weight: 0,
            weight_label: null_mut(),
            weight: null_mut(),
            p: 0.0,
            nu: 0.0,
            init_sol: null_mut(),
            regularize_bias: 0,
        };

        let owned_model = bindings::model {
            param: p,
            nr_class: num_classes,
            nr_feature: num_features,
            w: null_mut(),
            label: null_mut(),
            bias,
            rho: 0.0,
        };
        
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