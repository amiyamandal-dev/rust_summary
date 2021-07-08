extern crate anyhow;
use actix_web::middleware::Logger;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder, Result};
use env_logger::Env;
use rust_bert::bart::{
    BartConfigResources, BartMergesResources, BartModelResources, BartVocabResources,
};
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use rust_bert::resources::{RemoteResource, Resource};
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::time::Instant;
use tch::Device;

struct SummModel {
    summarization_model: SummarizationModel,
}

fn convert_vec_to_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

impl SummModel {
    fn new(summarization_config: SummarizationConfig) -> anyhow::Result<Self> {
        let summarization_model = SummarizationModel::new(summarization_config)?;
        Ok(SummModel {
            summarization_model,
        })
    }

    fn summerize_value(&mut self, input_string: String) -> anyhow::Result<String> {
        let input = [input_string.as_str()];
        let _output = self.summarization_model.summarize(&input);
        let joined = _output.join(" ");
        Ok(joined)
    }
}

fn summerization_for_single_input(
    min_length: i64,
    max_length: i64,
    data: String,
    num_beams: i64,
) -> String {
    /*
    this summrizer model
    */
    let now = Instant::now();
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartConfigResources::BART_CNN,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartVocabResources::BART_CNN,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartMergesResources::BART_CNN,
    ));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartModelResources::BART_CNN,
    ));
    let summarization_config = SummarizationConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        num_beams: num_beams,
        length_penalty: 2.0,
        min_length: min_length,
        max_length: max_length,
        device: Device::Cpu,
        ..Default::default()
    };
    let mut v = SummModel::new(summarization_config).unwrap();
    let rez = v.summerize_value(data).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    rez
}

fn summerization_for_multiple_input(
    min_length: i64,
    max_length: i64,
    data: Vec<String>,
    num_beams: i64,
) -> Vec<String> {
    /*
    this summrizer model
    */
    let now = Instant::now();
    let config_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartConfigResources::BART_CNN,
    ));
    let vocab_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartVocabResources::BART_CNN,
    ));
    let merges_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartMergesResources::BART_CNN,
    ));
    let model_resource = Resource::Remote(RemoteResource::from_pretrained(
        BartModelResources::BART_CNN,
    ));
    let summarization_config = SummarizationConfig {
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource,
        num_beams: num_beams,
        length_penalty: 2.0,
        min_length: min_length,
        max_length: max_length,
        device: Device::Cpu,
        ..Default::default()
    };
    let mut v = SummModel::new(summarization_config).unwrap();
    let mut result: Vec<String> = vec![];
    for i in data {
        result.push(v.summerize_value(i).unwrap());
    }

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    result
}

#[derive(Serialize, Deserialize, Clone)]
struct InputSumm {
    input_string: String,
    min_length: i64,
    max_length: i64,
    num_beams: i64,
}

#[derive(Serialize, Deserialize)]
struct OutputResponse {
    output_string: String,
}

#[get("/")]
async fn hello() -> impl Responder {
    HttpResponse::Ok().body("Hello world!")
}

#[post("/api/summ/single_input")]
async fn summerizer_api(data: web::Json<InputSumm>) -> Result<HttpResponse> {
    // this will summrize api for the input
    let var_name: &String = &data.input_string;
    let result = summerization_for_single_input(
        data.min_length,
        data.max_length,
        var_name.to_string(),
        data.num_beams,
    );

    Ok(HttpResponse::Ok().json(OutputResponse {
        output_string: result,
    }))
}

#[derive(Serialize, Deserialize, Clone)]
struct InputSummBatch {
    input_string: Vec<String>,
    min_length: i64,
    max_length: i64,
    num_beams: i64,
}

#[derive(Serialize, Deserialize)]
struct OutputResponseBatch {
    output_string: Vec<String>,
}
#[post("/api/summ/batch_input")]
async fn summerizer_api_batch(data: web::Json<InputSummBatch>) -> Result<HttpResponse> {
    // this will summrize api for the input
    let result = summerization_for_multiple_input(
        data.min_length,
        data.max_length,
        data.input_string.clone(),
        data.num_beams,
    );
    println!("{:?}", result);

    Ok(HttpResponse::Ok().json(OutputResponseBatch {
        output_string: result.clone(),
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    HttpServer::new(|| {
        App::new()
            .data(web::JsonConfig::default())
            .service(hello)
            .service(summerizer_api)
            .service(summerizer_api_batch)
    })
    .workers(4)
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
