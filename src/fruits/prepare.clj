(ns fruits.prepare
  (:require [mikera.image.core :as imagez]
            [mikera.image.filters :as filters]
            [clojure.string :as string]
            [clojure.java.io :as io]))

(defn preprocess-image
  "scale to image-size and convert the picture to grayscale"
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (println "> " img-path)
      (io/make-parents img-path)
      (try
      (-> (imagez/load-image file)
          ((filters/grayscale))
          ; this is bad, does not keep the ratio
          (imagez/resize image-size image-size)
          ; but this is not good either
          ; (imagez/resize image-size)
          (imagez/save img-path))
      (catch Exception e (println "Skip:" file " due to " (str e)))))))

(defn gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))

(defn- produce-indexed-data-label-seq
 [files]
 (->> (map (fn [file] [file (-> (.getName file) (string/split #"_") first)]) files)
      (map-indexed vector)))

; use first half of the files for training
; and second half for testing
(defn build-image-data
  [original-data-dir target-image-size]
  (let [
        training-dir (.getParent (io/as-file original-data-dir) "/training")
        testing-dir (.getParent (io/as-file original-data-dir) "/testing")
        
        files (gather-files original-data-dir)
        pfiles (partition (int (/ (count files) 2)) (shuffle files))
        training-observation-label-seq 
          (produce-indexed-data-label-seq (first pfiles))
        testing-observation-label-seq 
          (produce-indexed-data-label-seq (last pfiles))
        train-fn (partial preprocess-image training-dir target-image-size)
        test-fn (partial preprocess-image  testing-dir target-image-size)]
  (dorun (pmap train-fn training-observation-label-seq))
  (dorun (pmap test-fn training-observation-label-seq))))

(defn -main[& args]
  (if (empty? args)
    (println "Usage: lein run -m fruits.prepare <path-to-image-folder>")
    (fruits.prepare/build-image-data  (first args) 50)))

(comment

  (require '[fruits.prepare])
  (fruits.prepare/build-image-data  "fruits/original" 50)
  
)
