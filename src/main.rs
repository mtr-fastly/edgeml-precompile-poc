mod log;
mod ml;

use std::io::Cursor;
use std::ops::Deref;
use std::time::Instant;
use ::log::Level;
use fastly::http::{Method, StatusCode};
use fastly::{Error, Request, Response};
use log::emit_log;
use tract_flavour::prelude::{Datum, Framework, Graph, InferenceFact, InferenceModelExt, RunnableModel, tvec, TypedFact, TypedOp};
use once_cell::sync::Lazy;
use crate::log::emit_log_at_level;


pub static MODEL: Lazy<RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>> = Lazy::new(|| {

    // let model =
    tract_flavour::onnx()
        .model_for_read(&mut Cursor::new(include_bytes!("../models/mobilenet_v2.onnx"))).expect("Unable to read model.")
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224,224))).expect("Input fact error.")
        // .with_output_fact(0,InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 48, 112,112))).expect("Output fact error.")
        // Optimize the model.
        .into_optimized().expect("Optimization error.")
        // Make the model runnable and fix its inputs and outputs.
        .into_runnable().expect("Runnable transform failed.")
    // ;

    // model
});

#[export_name = "wizer.initialize"]
pub extern "C" fn init() {
    // Lazy::force(&MODEL);
}

#[fastly::main]
fn main(mut req: Request) -> Result<Response, Error> {

    // Log service version
    println!(
        "FASTLY_SERVICE_VERSION: {}",
        std::env::var("FASTLY_SERVICE_VERSION").unwrap_or_else(|_| String::new())
    );

    let mut resp = Response::new()
        .with_header("Access-Control-Allow-Origin", "*")
        .with_header("Access-Control-Allow-Headers", "Content-Type");
    let session = req.get_query_str().unwrap_or("session=")[8..].to_owned();
    let context = "main";
    match (req.get_method(), req.get_header_str("Content-Type")) {
        (&Method::POST, Some("image/jpeg")) => {
            emit_log(
                context,
                &session,
                "Loading model mobilenet_v2_1.4_224 (ImageNet).",
            );
            let inference_duration_start = Instant::now();
            let model = MODEL.deref();
            let model_loading_duration = inference_duration_start.elapsed().as_millis() as u64;
            emit_log(
                context,
                &session,
                &format!("Model loading took {model_loading_duration} ms!"),
            );

            match ml::infer(MODEL.deref(), &req.take_body_bytes(), &session) {
                Ok((confidence, label_index)) => {
                    let inference_duration = inference_duration_start.elapsed().as_millis() as u64;
                    emit_log(
                        context,
                        &session,
                        &format!("Image classified in {inference_duration} ms! ImageNet label index {} (confidence {:2}).", label_index, confidence),
                    );
                    resp.set_body_text_plain(&format!("{},{}", confidence, label_index));
                }
                Err(e) => {
                    emit_log(context, &session, &format!("Inference error: {:?}", e));
                    resp.set_body_text_plain(&format!("errored: {:?}", e));
                }
            }
        }
        (&Method::OPTIONS, _) => resp.set_status(StatusCode::OK),
        _ => resp.set_status(StatusCode::IM_A_TEAPOT),
    }

    Ok(resp)
}

#[test]
fn test_inference() {
    let session = "test";
    let context = "main";

    emit_log(context, session, "Starting inference test.");
    let start = Instant::now();
    Lazy::force(&MODEL);
    let elapsed = start.elapsed().as_millis();
    emit_log("precompilation", "model", &format!("Precompiling model took: {elapsed}"));
    let image = include_bytes!("../models/grace_hopper.jpg");
    let mut total_elapsed = 0u64;
    for i in 0..100000 {
        let inference_duration_start = Instant::now();
        match ml::infer(MODEL.deref(), image, &session) {
            Ok((confidence, label_index)) => {
                let inference_duration = inference_duration_start.elapsed().as_millis() as u64;
                total_elapsed += inference_duration;
                // emit_log_at_level(
                //     Level::Debug,
                //     context,
                //     session,
                //     &format!("Image classified in {inference_duration} ms! ImageNet label index {} (confidence {:2}).", label_index, confidence),
                // );
                if i % 100 == 0 {
                    emit_log(context, session, &format!("Average inference time over {i} runs: {}", total_elapsed as f64 / i as f64));
                }
            }
            Err(e) => {
                emit_log(context, &session, &format!("Inference error: {:?}", e));
            }
        }
    }
}