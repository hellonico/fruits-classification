(defproject fruits-classification "0.1"
  :description "An example of using experiment/classification on mnist."
  :repositories {"hellonico"   {:sign-releases false :url "https://repository.hellonico.info/repository/hellonico"}}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [thinktopic/experiment "0.9.22"]
                 

                 [org.clojure/tools.cli "0.3.5"]

                 ; to manipulate images
                 [thinktopic/think.image "0.4.8" ]
                 [thinktopic/think.datatype "0.3.17"]
                
                 [org.hellonico/imgscalr-lib "4.3"]
                 [org.hellonico/sizing "0.1.0"]
                 ; [net.mikera/imagez "0.12.0"]
                 ;;If you need cuda 8...
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 ;;If you need cuda 7.5...
                 ;;[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]
                 ]
  :exclusions [org.imgscalr/imgscalr-lib]

  ; :main mnist-classification.main
  ; :aot [mnist-classification.main]
  :jvm-opts ["-Xmx8000m"]
  :uberjar-name "classify-example.jar"

  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
