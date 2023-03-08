extern crate pyo3;
extern crate numpy;
extern crate rayon;

use std::cmp::Ord;
use std::cmp::Ordering;

use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::PyArray;


/// Compute pairwise Manhattan distances for a CSR sparse matrix.
#[pyfunction]
#[pyo3(signature = (data, indices, indptr, distances, threads=0))]
fn sparse_manhattan<'py>(
    py: Python<'py>,
    data: &PyArray<i32, numpy::Ix1>,
    indices: &PyArray<i32, numpy::Ix1>,
    indptr: &PyArray<i32, numpy::Ix1>,
    distances: &PyArray<f64, numpy::Ix1>,
    threads: usize,
) -> PyResult<()> {

    let indptr_r = indptr.try_readonly()?;
    let indices_r = indices.try_readonly()?;
    let data_r = data.try_readonly()?;
    
    let indptr_s = indptr_r.as_slice()?;
    let indices_s = indices_r.as_slice()?;
    let data_s = data_r.as_slice()?;
    
    let mut dist_r = distances.try_readwrite()?;
    let mut dist_s = dist_r.as_slice_mut()?.as_mut_ptr() as usize;
    let n = (distances.shape()[0] as f32 * 2.0).sqrt().ceil() as usize; 


    let pool = rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
        .expect("failed to create thread pool");
   
   
        
    py.allow_threads(|| {
        pool.install(|| {
            (0..n-1).into_par_iter()
                .map(|px| {
                    let i_next = indptr_s[px + 1] as usize;
                    for py in px+1..n {
                        let j_next = indptr_s[py + 1] as usize;

                        let mut i = indptr_s[px] as usize;
                        let mut j = indptr_s[py] as usize;

                        let mut d = 0;
                        while i < i_next && j < j_next {
                            match indices_s[i].cmp(&indices_s[j]) {
                                Ordering::Equal => {
                                    d += (data_s[i] - data_s[j]).abs();
                                    i += 1;
                                    j += 1;
                                }
                                Ordering::Less => {
                                    d += data_s[i].abs();
                                    i += 1;
                                }
                                Ordering::Greater => {
                                    d += data_s[j].abs();
                                    j += 1;
                                }
                            }
                        }

                        if i == i_next {
                            while j < j_next {
                                d += data_s[j].abs();
                                j += 1;
                            }
                        } else {
                            while i < i_next {
                                d += data_s[i].abs();
                                i += 1;
                            }
                        }

                        let k = n*px - px*(px+1)/2 + py - 1 - px;
                        unsafe {
                            let dist_ptr = dist_s as *mut f64;
                            *dist_ptr.add(k) = d as f64;
                        }
                    }
                })
                .collect::<()>();
        });
    });

    Ok(())
}




/// A Python module for metagenomic sequence comparison with ``skani``.
///
#[pymodule]
#[pyo3(name = "hca")]
pub fn init(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__package__", "htgcf")?;
    m.add_function(wrap_pyfunction!(sparse_manhattan, m)?)?;
    Ok(())
}
