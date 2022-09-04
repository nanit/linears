use linears::owned::OwnedModel;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct MatlabModelJson {
    pub model: MatlabModel,
}

#[derive(Debug, Deserialize)]
pub struct MatlabModel {
    #[serde(rename = "Parameters")]
    pub solver: i32,

    #[serde(rename = "nr_class")]
    pub num_classes: i32,

    #[serde(rename = "nr_feature")]
    pub num_features: i32,

    pub bias: f64,

    #[serde(rename = "Label")]
    pub labels: Vec<Vec<i32>>,

    pub w: Vec<f64>,
}

impl From<MatlabModelJson> for OwnedModel {
    fn from(mmj: MatlabModelJson) -> Self {
        let mm = mmj.model;
        let labels: Vec<i32> = mm.labels.iter().map(|v| *v.get(0).unwrap()).collect();
        let w = mm.w.clone();

        OwnedModel::new(mm.solver, mm.num_classes, mm.num_features, mm.bias, labels, w)
    }
}
