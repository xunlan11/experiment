
(cl:in-package :asdf)

(defsystem "hello_pkg-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Num" :depends-on ("_package_Num"))
    (:file "_package_Num" :depends-on ("_package"))
    (:file "homework_msg" :depends-on ("_package_homework_msg"))
    (:file "_package_homework_msg" :depends-on ("_package"))
  ))