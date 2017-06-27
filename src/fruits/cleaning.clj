(ns fruits.cleaning
	(:require [sizing.core :as sizing]))

(defn -main[& args]
  (let[path (first args) pngs (sizing/gather-png path) ]
  (sizing/check-sizes pngs)))
