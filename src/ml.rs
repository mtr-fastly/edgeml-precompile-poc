use std::io::Cursor;
use once_cell::sync::Lazy;
use tract_flavour::prelude::*;
use crate::log::emit_log;



// The inference function returns a tuple:
// (confidence, index of the predicted class)
pub fn infer(model: &RunnableModel<TypedFact,Box<dyn TypedOp>,Graph<TypedFact, Box<dyn TypedOp>>>, image_bytes: &[u8], session: &str) -> TractResult<(f32,i32)> {
    let context = "inference_engine";
    emit_log(
        context,
        session,
        "Optimizing runnable Tensorflow model for F32 datum type, tensor shape [1, 224, 224, 3].",
    );

    // Create a new image from the image byte slice.
    let img = image::load_from_memory(image_bytes)?.to_rgb8();
    emit_log(
        context,
        session,
        "Resizing image to fit 224x224 (filter algorithm: nearest neighbour).",
    );
    // Resize the input image to the dimension the model was trained on.
    // Sampling filter and performance comparison: https://docs.rs/image/0.23.12/image/imageops/enum.FilterType.html#examples
    // Switch to FilterType::Triangle if you're getting odd results.
    let resized = image::imageops::resize(&img, 224, 224, image::imageops::FilterType::Nearest);

    emit_log(
        context,
        session,
        "Converting scaled image to tensor and running model...",
    );
    // Make a Tensor out of it.
    let img: Tensor = tract_ndarray::Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    })
    .into_tensor();

    // Run the model on the input.
    let result = model.run(tvec!(img.into()))?;
    emit_log(
        context,
        session,
        &format!("Inference complete. Traversing results graph to find a best-confidence fit...")
    );

    // Find the max value with its index.
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
    Ok(best.unwrap())
}
