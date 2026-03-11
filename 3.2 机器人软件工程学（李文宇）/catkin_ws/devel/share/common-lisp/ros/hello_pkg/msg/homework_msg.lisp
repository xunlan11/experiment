; Auto-generated. Do not edit!


(cl:in-package hello_pkg-msg)


;//! \htmlinclude homework_msg.msg.html

(cl:defclass <homework_msg> (roslisp-msg-protocol:ros-message)
  ((number
    :reader number
    :initarg :number
    :type cl:float
    :initform 0.0)
   (text
    :reader text
    :initarg :text
    :type cl:string
    :initform ""))
)

(cl:defclass homework_msg (<homework_msg>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <homework_msg>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'homework_msg)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hello_pkg-msg:<homework_msg> is deprecated: use hello_pkg-msg:homework_msg instead.")))

(cl:ensure-generic-function 'number-val :lambda-list '(m))
(cl:defmethod number-val ((m <homework_msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hello_pkg-msg:number-val is deprecated.  Use hello_pkg-msg:number instead.")
  (number m))

(cl:ensure-generic-function 'text-val :lambda-list '(m))
(cl:defmethod text-val ((m <homework_msg>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hello_pkg-msg:text-val is deprecated.  Use hello_pkg-msg:text instead.")
  (text m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <homework_msg>) ostream)
  "Serializes a message object of type '<homework_msg>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'number))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'text))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'text))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <homework_msg>) istream)
  "Deserializes a message object of type '<homework_msg>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'number) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'text) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'text) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<homework_msg>)))
  "Returns string type for a message object of type '<homework_msg>"
  "hello_pkg/homework_msg")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'homework_msg)))
  "Returns string type for a message object of type 'homework_msg"
  "hello_pkg/homework_msg")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<homework_msg>)))
  "Returns md5sum for a message object of type '<homework_msg>"
  "9ae0a379b0fd3f8da3d0c153afea9869")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'homework_msg)))
  "Returns md5sum for a message object of type 'homework_msg"
  "9ae0a379b0fd3f8da3d0c153afea9869")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<homework_msg>)))
  "Returns full string definition for message of type '<homework_msg>"
  (cl:format cl:nil "float32 number~%string text~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'homework_msg)))
  "Returns full string definition for message of type 'homework_msg"
  (cl:format cl:nil "float32 number~%string text~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <homework_msg>))
  (cl:+ 0
     4
     4 (cl:length (cl:slot-value msg 'text))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <homework_msg>))
  "Converts a ROS message object to a list"
  (cl:list 'homework_msg
    (cl:cons ':number (number msg))
    (cl:cons ':text (text msg))
))
