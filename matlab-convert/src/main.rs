mod matlab;

use crate::matlab::MatlabModel;
use linears::owned::OwnedModel;
use matlab::MatlabModelJson;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;

fn main() {
    let mat_json =
        fs::read("../sleep_awake/statemachine/liblinear_models/correctSVMBeginModel.json").unwrap();
    let model: MatlabModel = serde_json::from_slice(&mat_json).unwrap();
    
    let owned = OwnedModel::from(MatlabModelJson { model });
    let model = owned.model();
    // owned.model().save("model7.bin").unwrap();

    // let model = Model::load_from_file("model7.bin").unwrap();

    let features_file =
        fs::read("../sleep_awake/statemachine/liblinear_models/feratureMatrixBegin.json").unwrap();
    let features: Vec<Vec<f64>> = serde_json::from_slice(&features_file).unwrap();

    let preds: HashMap<usize, f64> = features
        .par_iter()
        .enumerate()
        .map(|(r_idx, r)| {
            (r_idx, model.predict(r))
        })
        .collect();

    println!("{preds:?}");
}
