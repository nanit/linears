use linears::{bindings, Model, OwnedModel};
use serde::Deserialize;
use std::ptr::{null, null_mut};

#[derive(Debug, Deserialize)]
pub struct MatlabModelJson {
    pub model: MatlabModel,
}

#[derive(Debug, Deserialize)]
pub struct MatlabModel {
    #[serde(rename = "Parameters")]
    solver: i32,

    #[serde(rename = "nr_class")]
    num_classes: i32,

    #[serde(rename = "nr_feature")]
    num_features: i32,

    bias: f64,

    #[serde(rename = "Label")]
    labels: Vec<Vec<i32>>,

    w: Vec<f64>,
}

impl From<MatlabModelJson> for OwnedModel {
    fn from(mmj: MatlabModelJson) -> Self {
        let mm = mmj.model;

        let p = bindings::parameter {
            solver_type: mm.solver,

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

        let labels: Vec<i32> = mm.labels.iter().map(|v| *v.get(0).unwrap()).collect();
        let w = mm.w.clone();
        let owned_model = bindings::model {
            param: p,
            nr_class: mm.num_classes,
            nr_feature: mm.num_features,
            w: null_mut(),
            label: null_mut(),
            bias: mm.bias,
            rho: 0.0,
        };

        OwnedModel::new(owned_model, labels, w)
    }
}
