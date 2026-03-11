;; Auto-generated. Do not edit!


(when (boundp 'hello_pkg::homework_msg)
  (if (not (find-package "HELLO_PKG"))
    (make-package "HELLO_PKG"))
  (shadow 'homework_msg (find-package "HELLO_PKG")))
(unless (find-package "HELLO_PKG::HOMEWORK_MSG")
  (make-package "HELLO_PKG::HOMEWORK_MSG"))

(in-package "ROS")
;;//! \htmlinclude homework_msg.msg.html


(defclass hello_pkg::homework_msg
  :super ros::object
  :slots (_number _text ))

(defmethod hello_pkg::homework_msg
  (:init
   (&key
    ((:number __number) 0.0)
    ((:text __text) "")
    )
   (send-super :init)
   (setq _number (float __number))
   (setq _text (string __text))
   self)
  (:number
   (&optional __number)
   (if __number (setq _number __number)) _number)
  (:text
   (&optional __text)
   (if __text (setq _text __text)) _text)
  (:serialization-length
   ()
   (+
    ;; float32 _number
    4
    ;; string _text
    4 (length _text)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _number
       (sys::poke _number (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; string _text
       (write-long (length _text) s) (princ _text s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _number
     (setq _number (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; string _text
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _text (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(setf (get hello_pkg::homework_msg :md5sum-) "9ae0a379b0fd3f8da3d0c153afea9869")
(setf (get hello_pkg::homework_msg :datatype-) "hello_pkg/homework_msg")
(setf (get hello_pkg::homework_msg :definition-)
      "float32 number
string text

")



(provide :hello_pkg/homework_msg "9ae0a379b0fd3f8da3d0c153afea9869")


