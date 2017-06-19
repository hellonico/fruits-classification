(ns fruits.training
  (require 
    [clojure.java.io :as io]
    [think.image.patch :as patch]
    [mikera.image.core :as imagez]
    [cortex.experiment.util :as experiment-util]
    [cortex.experiment.classification :as classification]
    [cortex.nn.layers :as layers]))

(defn initial-description-preset [input-w input-h num-classes] 
    [(layers/input input-w input-h 1 :id :data)
     (layers/convolutional 5 0 1 20)
     (layers/max-pooling 2 0 2)
     (layers/dropout 0.9)
     (layers/relu)
     (layers/convolutional 5 0 1 50)
     (layers/max-pooling 2 0 2)
     (layers/batch-normalization)
     (layers/linear 1000)
     (layers/relu  :center-loss {:label-indexes {:stream :labels}
                   :label-inverse-counts {:stream :labels}
                   :labels {:stream :labels}
                   :alpha 0.9
                   :lambda 1e-4})
     (layers/dropout 0.5)
     (layers/linear num-classes)
     (layers/softmax :id :labels)])

(defn observation->image [ image-size observation] 
  (patch/patch->image observation image-size))

(defn training[base-folder]
 (let[  
  dataset-folder  base-folder
  training-folder (str dataset-folder "training")
  test-folder     (str dataset-folder "testing")
  categories      (into [] (map #(.getName %) (.listFiles (io/file training-folder))))
  num-classes     (count categories)
  class-mapping
      {:class-name->index (zipmap categories (range))
       :index->class-name (zipmap (range) categories)}

  first-test-pic  (imagez/load-image (first (filter #(.isFile %) (file-seq (io/file training-folder)))))
  image-size      (.getWidth (imagez/load-image first-test-pic))

    train-ds
    (-> training-folder
      (experiment-util/create-dataset-from-folder
      class-mapping :image-aug-fn (:image-aug-fn {}))
    (experiment-util/infinite-class-balanced-dataset))

  test-ds
    (-> test-folder
     (experiment-util/create-dataset-from-folder class-mapping))

  initial-description (initial-description-preset image-size image-size num-classes)
    
  ]
  (classification/perform-experiment
     initial-description
     train-ds
     test-ds
     (partial observation->image image-size)
     class-mapping
     {}) ))

(defn -main[& args]
  (if (empty? args)
    (println "Usage: lein run -m fruits.training <path-to-folder-containing-training-and-testing-folders>")    
    (fruits.training/training (first args))))

(comment

  (require '[fruits.training])
  (fruits.training/training "fruits/")

  )