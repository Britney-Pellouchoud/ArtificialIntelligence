(define (caar x) (car (car x)))
(define (cadr x) (car (cdr x)))
(define (cdar x) (cdr (car x)))
(define (cddr x) (cdr (cdr x)))

; Some utility functions that you may find useful to implement.

(define (cons-all first rests)
  'replace-this-line)

(define range
  (lambda (n m)
    (cond
      ((= n m) '())
        (else (cons n (range ((if (< n m) + -) n 1) m))))))
  

(define (zip lsts)
  (cond
    ((null? lsts) '(() ()))
    ((null? (car lsts)) '())
    (else 
    (begin 
    (define first (map (lambda (x) (car x)) lsts))
    (define second (map (lambda (x) (cdr x)) lsts))
    (cons first (zip second))
    )
    ) 
  )
)

;; Problem 17
;; Returns a list of two-element lists

(define (enumerate s)
  ; BEGIN PROBLEM 17
  (zip (list (range 0 (length s)) s))
  )
  ; END PROBLEM 17

;; Problem 18
;; List all ways to make change for TOTAL with DENOMS
(define (list-change total denoms)
  ; BEGIN PROBLEM 18
  (define (helper total denoms lst)
    (cond
      ((= total 0) (list lst)
      )
      ((< total 0)
        '())
      ((null? denoms) '())
      (else
        (append (helper (- total (car denoms)) denoms (append lst (list (car denoms))))
          (helper total (cdr denoms) lst)) 

          )
        


    )
  )
  (helper total denoms '()))

  
  
  ; END PROBLEM 18

;; Problem 19
;; Returns a function that checks if an expression is the special form FORM
(define (check-special form)
  (lambda (expr) (equal? form (car expr))))

(define lambda? (check-special 'lambda))
(define define? (check-special 'define))
(define quoted? (check-special 'quote))
(define let?    (check-special 'let))

;; Converts all let special forms in EXPR into equivalent forms using lambda
(define (let-to-lambda expr)
  (cond ((atom? expr)
         ; BEGIN PROBLEM 19
         expr
         ; END PROBLEM 19
         )
        ((quoted? expr)
         ; BEGIN PROBLEM 19
         expr

         ; END PROBLEM 19
         )
        ((or (lambda? expr)
             (define? expr))
         (let ((form   (car expr))
               (params (cadr expr))
               (body   (cddr expr)))
           ; BEGIN PROBLEM 19
           (cons
            form 
            (cons params
              (let-to-lambda body))
              )
            )
           
            )
           ; END PROBLEM 19
           

        ((let? expr)
           (let ((values (cadr expr))
                 (body   (cddr expr))) 
           ; BEGIN PROBLEM 19
            ;(print (zip values))
            ;(print body)
            ;(print (cons 
                ;'lambda (car (zip values) (car body)) nil) )
            ;(print (cons 'lambda values))
            (cons 
              (cons 
                'lambda 
                (cons 
                  (let-to-lambda (car (zip values))) 
                  (let-to-lambda body)) )
              
                (let-to-lambda (cadr (zip values))) 
                )

           ; END PROBLEM 19
           )
           )
        (else
         ; BEGIN PROBLEM 19
         ;(print ***************)
         ;(print (cdr (car expr)))
         (map let-to-lambda expr)
         ;(print (cdr expr))
         ; END PROBLEM 19
         )
         ))  
