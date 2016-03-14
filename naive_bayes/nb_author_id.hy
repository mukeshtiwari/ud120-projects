(import [sys]
        [time [time]]
        [sklearn.naive_bayes [GaussianNB]]
        [sklearn.metrics [accuracy_score]])
(sys.path.append "../tools")
(import [email_preprocess [preprocess]])



(defn predfunction []
  (do
   (def (, features_train features_test labels_train labels_test)
     (preprocess))
   (def pre (.predict (.fit (GaussianNB) features_train labels_train)
                      features_test))
   (accuracy_score labels_test pre)))

(print (predfunction))
