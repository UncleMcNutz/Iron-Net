//! # Multi-Modal Concepts: Images, Text, and Sound as 3D Forms
//!
//! If iron thinks in 3D configurations, then ALL modalities can map there.
//! - Text: words → forms (already built)
//! - Images: pixels → forms (spatial naturally)
//! - Sound: waveforms → forms (frequency as geometry)
//!
//! The same iron substrate processes all modalities.
//! Translation between modalities = finding similar forms.
//!
//! ## The Unification
//!
//! Different inputs, same thinking substrate:
//!
//! ```text
//!   [Image] ──→ Encoder ──╮
//!                         │
//!   [Text]  ──→ Encoder ──┼──→ [3D Form] ──→ Iron Thinks ──→ [3D Form]
//!                         │
//!   [Sound] ──→ Encoder ──╯
//! ```
//!
//! A picture of a cat and the word "cat" map to similar 3D forms.

use crate::concept::{Concept, ConceptSculptor};
use crate::spin::Spin;

/// Modality type
#[derive(Clone, Debug, PartialEq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Numeric,
}

/// Multi-modal encoder: converts various inputs to 3D concept forms
pub struct MultiModalEncoder {
    /// Size of concept lattice
    pub size: usize,
    sculptor: ConceptSculptor,
}

impl MultiModalEncoder {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            sculptor: ConceptSculptor::new(size, size, size),
        }
    }
    
    // ========================
    // IMAGE → 3D FORM
    // ========================
    
    /// Encode grayscale image to 3D form
    /// Image becomes xy slice, depth encodes intensity
    pub fn encode_image(&self, pixels: &[f32], width: usize, height: usize) -> Concept {
        let mut concept = Concept::empty(self.size, self.size, self.size);
        
        for y in 0..height.min(self.size) {
            for x in 0..width.min(self.size) {
                let pixel_idx = y * width + x;
                if pixel_idx >= pixels.len() { continue; }
                
                let intensity = pixels[pixel_idx].clamp(0.0, 1.0);
                
                // Map intensity to depth (brighter = taller column)
                let max_z = (intensity * self.size as f32) as usize;
                for z in 0..max_z {
                    concept.set(x, y, z, Spin::Up);
                }
            }
        }
        
        concept.with_label("image")
    }
    
    /// Encode binary image (black/white) as single slice
    pub fn encode_binary_image(&self, pixels: &[bool], width: usize, height: usize) -> Concept {
        let mut concept = Concept::empty(self.size, self.size, self.size);
        let mid_z = self.size / 2;
        
        for y in 0..height.min(self.size) {
            for x in 0..width.min(self.size) {
                let pixel_idx = y * width + x;
                if pixel_idx < pixels.len() && pixels[pixel_idx] {
                    // Spread the pixel across a few z layers for thickness
                    for dz in 0..3 {
                        let z = (mid_z + dz).min(self.size - 1);
                        concept.set(x, self.size - 1 - y, z, Spin::Up);  // Flip y for visual
                    }
                }
            }
        }
        
        concept.with_label("binary_image")
    }
    
    /// Decode 3D form back to image (take middle z-slice)
    pub fn decode_to_image(&self, concept: &Concept) -> Vec<f32> {
        let (dx, dy, dz) = concept.dims;
        let mut pixels = vec![0.0f32; dx * dy];
        
        for y in 0..dy {
            for x in 0..dx {
                // Count how many z-layers are Up
                let mut count = 0;
                for z in 0..dz {
                    if concept.get(x, y, z) == Spin::Up {
                        count += 1;
                    }
                }
                pixels[y * dx + x] = count as f32 / dz as f32;
            }
        }
        
        pixels
    }
    
    // ========================
    // AUDIO → 3D FORM
    // ========================
    
    /// Encode audio waveform to 3D form
    /// X = time, Y = frequency bands, Z = amplitude
    pub fn encode_audio(&self, samples: &[f32], sample_rate: usize) -> Concept {
        let mut concept = Concept::empty(self.size, self.size, self.size);
        
        // Simple approach: chunk samples into time bins
        let samples_per_bin = samples.len().max(1) / self.size;
        
        for x in 0..self.size {
            let start = x * samples_per_bin;
            let end = ((x + 1) * samples_per_bin).min(samples.len());
            
            if start >= samples.len() { break; }
            
            // Get samples in this time window
            let chunk = &samples[start..end];
            if chunk.is_empty() { continue; }
            
            // Compute simple frequency bands via zero-crossing rate and amplitude
            let amplitude = chunk.iter().map(|s| s.abs()).sum::<f32>() / chunk.len() as f32;
            let zero_crossings = chunk.windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count();
            let zcr = zero_crossings as f32 / chunk.len() as f32;
            
            // Y = frequency (from zero-crossing rate)
            let freq_y = (zcr * self.size as f32 * 10.0) as usize % self.size;
            
            // Z = amplitude
            let amp_z = (amplitude * self.size as f32).min(self.size as f32 - 1.0) as usize;
            
            // Fill a region
            for dy in 0..3 {
                for dz in 0..=amp_z {
                    let y = (freq_y + dy).min(self.size - 1);
                    concept.set(x, y, dz, Spin::Up);
                }
            }
        }
        
        concept.with_label("audio")
    }
    
    /// Generate simple waveform from concept
    pub fn decode_to_audio(&self, concept: &Concept, num_samples: usize) -> Vec<f32> {
        let (dx, dy, dz) = concept.dims;
        let mut samples = vec![0.0f32; num_samples];
        let samples_per_x = num_samples / dx;
        
        for x in 0..dx {
            // Find dominant y (frequency) and z (amplitude) for this x slice
            let mut total_y = 0.0f32;
            let mut total_z = 0.0f32;
            let mut count = 0.0f32;
            
            for y in 0..dy {
                for z in 0..dz {
                    if concept.get(x, y, z) == Spin::Up {
                        total_y += y as f32;
                        total_z += z as f32;
                        count += 1.0;
                    }
                }
            }
            
            if count > 0.0 {
                let freq = total_y / count / dy as f32;  // 0-1
                let amp = total_z / count / dz as f32;   // 0-1
                
                // Generate sinusoid for this time segment
                let base_freq = 200.0 + freq * 800.0;  // 200-1000 Hz
                for i in 0..samples_per_x {
                    let sample_idx = x * samples_per_x + i;
                    if sample_idx < num_samples {
                        let t = sample_idx as f32 / 44100.0;  // Assume 44.1kHz
                        samples[sample_idx] = amp * (2.0 * std::f32::consts::PI * base_freq * t).sin();
                    }
                }
            }
        }
        
        samples
    }
    
    // ========================
    // NUMERIC → 3D FORM
    // ========================
    
    /// Encode a vector of numbers to 3D form
    pub fn encode_vector(&self, values: &[f32]) -> Concept {
        let mut concept = Concept::empty(self.size, self.size, self.size);
        
        // Map each value to a column
        for (i, &v) in values.iter().take(self.size * self.size).enumerate() {
            let x = i % self.size;
            let y = i / self.size;
            
            let normalized = (v.clamp(-1.0, 1.0) + 1.0) / 2.0;  // Map to 0-1
            let height = (normalized * self.size as f32) as usize;
            
            for z in 0..height {
                concept.set(x, y, z, Spin::Up);
            }
        }
        
        concept.with_label("vector")
    }
    
    /// Decode 3D form to vector
    pub fn decode_to_vector(&self, concept: &Concept) -> Vec<f32> {
        let (dx, dy, dz) = concept.dims;
        let mut values = Vec::with_capacity(dx * dy);
        
        for y in 0..dy {
            for x in 0..dx {
                let mut height = 0;
                for z in 0..dz {
                    if concept.get(x, y, z) == Spin::Up {
                        height = z + 1;
                    }
                }
                let normalized = height as f32 / dz as f32;
                let value = normalized * 2.0 - 1.0;  // Map back to -1 to 1
                values.push(value);
            }
        }
        
        values
    }
    
    // ========================
    // CROSS-MODAL
    // ========================
    
    /// Create a multi-modal concept that combines multiple inputs
    pub fn fuse_modalities(&self, concepts: &[Concept]) -> Concept {
        if concepts.is_empty() {
            return Concept::empty(self.size, self.size, self.size);
        }
        
        // Superimpose all concepts with averaging
        let n = concepts.len() as f32;
        let mut form = vec![0.0f32; self.size * self.size * self.size];
        
        for concept in concepts {
            for (i, &spin) in concept.form.iter().enumerate() {
                form[i] += spin.value() / n;
            }
        }
        
        // Threshold to binary
        let result_form: Vec<Spin> = form.iter()
            .map(|&v| if v > 0.0 { Spin::Up } else { Spin::Down })
            .collect();
        
        Concept {
            form: result_form,
            dims: (self.size, self.size, self.size),
            label: Some("fused".to_string()),
        }
    }
}

/// Simple ASCII art to binary image
pub fn ascii_to_pixels(ascii: &str, width: usize, height: usize) -> Vec<bool> {
    let chars: Vec<char> = ascii.chars().filter(|c| !c.is_whitespace() || *c == ' ').collect();
    let mut pixels = vec![false; width * height];
    
    for (i, &c) in chars.iter().take(width * height).enumerate() {
        pixels[i] = c != ' ' && c != '.';
    }
    
    pixels
}

/// Demonstrate multi-modal encoding
pub fn demo_multimodal() {
    use crate::concept::visualize_concept;
    
    println!("================================================================");
    println!("      MULTI-MODAL: Images, Audio, Numbers as 3D Forms");
    println!("================================================================");
    println!();
    
    let size = 10;
    let encoder = MultiModalEncoder::new(size);
    
    // ========================
    // IMAGE TO CONCEPT
    // ========================
    println!("=== IMAGE → 3D FORM ===\n");
    
    // Simple smiley face as ASCII
    let smiley = "\
        ..####..\
        .#....#.\
        #.#..#.#\
        #......#\
        #.#..#.#\
        #..##..#\
        .#....#.\
        ..####..";
    
    println!("Input image (8x8):");
    for row in 0..8 {
        print!("  ");
        for col in 0..8 {
            let c = smiley.chars().nth(row * 8 + col).unwrap_or(' ');
            print!("{}", if c == '#' { "██" } else { "  " });
        }
        println!();
    }
    
    let pixels = ascii_to_pixels(smiley, 8, 8);
    let image_concept = encoder.encode_binary_image(&pixels, 8, 8);
    
    println!("\nAs 3D concept:");
    visualize_concept(&image_concept);
    
    // ========================
    // AUDIO TO CONCEPT
    // ========================
    println!("\n=== AUDIO → 3D FORM ===\n");
    
    // Generate a simple tone
    let sample_rate = 44100;
    let duration = 0.1;
    let num_samples = (sample_rate as f32 * duration) as usize;
    
    let mut samples = vec![0.0f32; num_samples];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        // 440 Hz sine wave with envelope
        let envelope = (t * 10.0).min(1.0) * (1.0 - t / duration).max(0.0);
        samples[i] = envelope * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
    }
    
    println!("Input: 440 Hz sine wave (0.1s)");
    println!("Samples: {}", num_samples);
    
    let audio_concept = encoder.encode_audio(&samples, sample_rate);
    println!("\nAs 3D concept:");
    visualize_concept(&audio_concept);
    
    // ========================
    // VECTOR TO CONCEPT
    // ========================
    println!("\n=== NUMERIC VECTOR → 3D FORM ===\n");
    
    let vector = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0];
    println!("Input vector: {:?}", vector);
    
    let vector_concept = encoder.encode_vector(&vector);
    println!("\nAs 3D concept:");
    visualize_concept(&vector_concept);
    
    // Decode back
    let decoded = encoder.decode_to_vector(&vector_concept);
    println!("\nDecoded back: {:?}", decoded.iter().take(10).map(|v| format!("{:.1}", v)).collect::<Vec<_>>());
    
    // ========================
    // FUSION
    // ========================
    println!("\n=== MULTI-MODAL FUSION ===\n");
    
    println!("Fusing image + audio + vector into single concept...\n");
    let fused = encoder.fuse_modalities(&[
        image_concept.clone(),
        audio_concept.clone(),
        vector_concept.clone(),
    ]);
    visualize_concept(&fused);
    
    // ========================
    // CROSS-MODAL SIMILARITY
    // ========================
    println!("\n=== CROSS-MODAL SIMILARITY ===\n");
    
    println!("Can different modalities produce similar concepts?");
    println!();
    
    // Create a "rising" pattern in different modalities
    
    // Rising image (diagonal)
    let mut rising_pixels = vec![false; 64];
    for i in 0..8 {
        rising_pixels[i * 8 + i] = true;
        if i > 0 { rising_pixels[i * 8 + i - 1] = true; }
        if i < 7 { rising_pixels[i * 8 + i + 1] = true; }
    }
    let rising_image = encoder.encode_binary_image(&rising_pixels, 8, 8);
    
    // Rising audio (frequency sweep)
    let mut rising_audio = vec![0.0f32; 4410];
    for i in 0..4410 {
        let t = i as f32 / 44100.0;
        let freq = 200.0 + (i as f32 / 4410.0) * 800.0;  // 200 -> 1000 Hz
        rising_audio[i] = (2.0 * std::f32::consts::PI * freq * t).sin();
    }
    let rising_audio_concept = encoder.encode_audio(&rising_audio, 44100);
    
    // Rising vector
    let rising_vector: Vec<f32> = (0..10).map(|i| i as f32 / 10.0).collect();
    let rising_vector_concept = encoder.encode_vector(&rising_vector);
    
    println!("Image (diagonal) ~ Audio (sweep): {:.1}%", 
             rising_image.similarity(&rising_audio_concept) * 100.0);
    println!("Image (diagonal) ~ Vector (rising): {:.1}%",
             rising_image.similarity(&rising_vector_concept) * 100.0);
    println!("Audio (sweep) ~ Vector (rising): {:.1}%",
             rising_audio_concept.similarity(&rising_vector_concept) * 100.0);
    
    println!("\n=== THE INSIGHT ===\n");
    println!("All modalities map to the same 3D concept space.");
    println!("Iron doesn't know if input was image, sound, or numbers.");
    println!("It only knows FORM. The geometry is what matters.");
    println!();
    println!("A rising diagonal, a rising frequency, and a rising value");
    println!("all produce similar 3D configurations.");
    println!("That's the same thought, in different languages.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_image_encode_decode() {
        let encoder = MultiModalEncoder::new(8);
        
        let pixels: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let concept = encoder.encode_image(&pixels, 8, 8);
        let decoded = encoder.decode_to_image(&concept);
        
        // Should preserve rough gradient
        assert!(decoded[0] < decoded[63]);
    }
    
    #[test]
    fn test_vector_roundtrip() {
        let encoder = MultiModalEncoder::new(8);
        
        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let concept = encoder.encode_vector(&input);
        let output = encoder.decode_to_vector(&concept);
        
        // First 5 values should be roughly preserved
        for i in 0..5 {
            let expected = input[i] * 2.0 - 1.0;  // Map to -1 to 1
            assert!((output[i] - expected).abs() < 0.3);
        }
    }
}
