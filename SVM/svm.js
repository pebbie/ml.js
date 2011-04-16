/*
 * Support Vector Machine Library for javascript
 * ported from libsvm 3.1.0 by Peb Ruswono Aryan
 * 
 * TODO: 
 *  - implement super
 *  - parse model file
 *
 */
(function(){

this.SVM = null;

(function(){
    SVM = function()
    {
        //entry point
    };
    
    SVM.version = "310";
    
    SVM._new = function(defval){
        if (typeof(defval)=="function")
            return new defval();
        else if(typeof(defval)=="object")
            return SVM.clone(defval);
        else
            return defval;
    }
    
    SVM.clone = function(obj){
        var result;
        if(obj.length != undefined)
            result = [];
         else
            result = {};
        for(a in obj){
            if(typeof(obj[a])=="object")
                result[a] = SVM.clone(obj[a]);
            else
                result[a] = obj[a];
        }
        return result;
    }
    
    SVM.arr = function(dim, defval){
        var result = new Array();
        if(typeof(defval)=="object" && (defval.length != undefined)){
            if(typeof(dim)=="number"){
                for(var i=0; i<dim; ++i)
                    result[i] = SVM._new(defval[0]);
            }
            else if (dim.length == 1){
                for(var i=0; i<dim[0]; ++i)
                    result[i] = SVM._new(defval[0]);
            }
            else{
                for(var i=0; i<dim[0]; ++i)
                    result[i] = SVM.arr(dim.slice(1), defval.slice(1));
            }
        }
        else {
            if(typeof(dim)=="number"){
                for(var i=0; i<dim; ++i)
                    result[i] = SVM._new(defval);
            }
            else if (dim.length == 1){
                for(var i=0; i<dim[0]; ++i)
                    result[i] = SVM._new(defval);
            }
            else{
                for(var i=0; i<dim[0]; ++i)
                    result[i] = SVM.arr(dim.slice(1), defval);
            }
        }
        return result;
    }
    
    SVM.nextInt = function(range)
    {
        return Math.floor(Math.random()*range);
    }
    
    function tanh (arg) {
        // Returns the hyperbolic tangent of the number, defined as sinh(number)/cosh(number)  
        // 
        // version: 1103.1210
        // discuss at: http://phpjs.org/functions/tanh
        // +   original by: Onno Marsman
        // *     example 1: tanh(5.4251848798444815);
        // *     returns 1: 0.9999612058841574
        return (Math.exp(arg) - Math.exp(-arg)) / (Math.exp(arg) + Math.exp(-arg));
    }
    
    /* cache */
    
    SVM.head_t = function()
    {
        this.prev = null;
        this.next = null;
        this.data = null;
        this.len = 0;
    }
    
    SVM.Cache = function(l, size)
    {
        this.l = l;
        this.size = size;
        this.head = SVM.arr(l, SVM.head_t);
        this.lru_head = new SVM.head_t();
        this.lru_head.prev = this.lru_head;
        this.lru_head.next = this.lru_head;
        this.size = Math.floor(size/4);
        this.size -= l * Math.floor(size/4);
        this.size = Math.max(size, 2*l);
    }
    
    SVM.Cache.prototype = {
        lru_insert:function(h){
            h.next = this.lru_head;
            h.prev = this.lru_head.prev;
            h.prev.next = h;
            h.next.prev = h;
        },
        lru_delete:function(h){
            h.prev.next = h.next;
            h.next.prev = h.prev;
        },
        // request data [0,len)
        // return some position p where [p,len) need to be filled
        // (p >= len if nothing needs to be filled)
        // java: simulate pointer using single-element array
        get_data: function(index, data, len)
        {
            var h = this.head[index];
            if(h.len > 0) this.lru_delete(h);
            var more = len - h.len;

            if(more > 0){
                // free old space
                while(this.size < more){
                    var old = this.lru_head.next;
                    this.lru_delete(old);
                    this.size += old.len;
                    old.data = null;
                    old.len = 0;
                }

                // allocate new space
                var new_data = SVM.arr(len, 0.0);
                //if(h.data != null) System.arraycopy(h.data,0,new_data,0,h.len);
                if (h.data != null) new_data = SVM.clone(h.data);
                
                h.data = new_data;
                this.size -= more;
                _=h.len; h.len=len; len=_;
            }

            this.lru_insert(h);
            data[0] = h.data;
            return len;
        },
        swap_index: function(i, j){
            if(i==j) return;
            
            if(this.head[i].len > 0) this.lru_delete(head[i]);
            if(this.head[j].len > 0) this.lru_delete(head[j]);
            _=this.head[i].data; this.head[i].data=this.head[j].data; this.head[j].data=_;
            _=this.head[i].len; this.head[i].len=this.head[j].len; this.head[j].len=_;
            if(this.head[i].len > 0) this.lru_insert(this.head[i]);
            if(this.head[j].len > 0) this.lru_insert(this.head[j]);

            if(i>j) {_=i; i=j; j=_;}
            for(h = this.lru_head.next; h!=this.lru_head; h=h.next){
                if(h.len > i){
                    if(h.len > j) {
                        _=h.data[i]; h.data[i]=h.data[j]; h.data[j]=_;
                    }
                    else {
                        // give up
                        this.lru_delete(h);
                        size += h.len;
                        h.data = null;
                        h.len = 0;
                    }
                }
            }
        }
    }
    
    /* svm_node Class */
    SVM.Node = function(){
        this.index = 0;
        this.value = 0.0;
    }
    
    /* svm_parameter Class */
    SVM.Parameter = function(param)
    {
        this.svm_type  = SVM.C_SVC;
        this.kernel_type = SVM.LINEAR;
        this.degree = 3.0;    // for poly
        this.gamma = 0.0;    // for poly/rbf/sigmoid
        this.coef0 = 0.0;    // for poly/sigmoid
        
        // these are for training only
        this.cache_size = 100; // in MB
        this.eps = 1e-3;    // stopping criteria
        this.C = 0.0;    // for C_SVC, EPSILON_SVR and NU_SVR
        this.nr_weight = 0;        // for C_SVC
        this.weight_label = null;    // for C_SVC
        this.weight = null;        // for C_SVC
        this.nu = 0.5;    // for NU_SVC, ONE_CLASS, and NU_SVR
        this.p = 0.1;    // for EPSILON_SVR
        this.shrinking = 1;    // use the shrinking heuristics
        this.probability = 0; // do probability estimates
    }
    
    /* svm_type */
    SVM.C_SVC = 0;
    SVM.NU_SVC = 1;
    SVM.ONE_CLASS = 2;
    SVM.EPSILON_SVR = 3;
    SVM.NU_SVR = 4;
    
    /* kernel_type */
    SVM.LINEAR = 0;
    SVM.POLY = 1;
    SVM.RBF = 2;
    SVM.SIGMOID = 3;
    SVM.PRECOMPUTED = 4;

    /* svm_problem Class */
    SVM.Problem = function(y_, x_)
    {
        this.l = 0;
        this.y = y_ || null;
        this.x = x_ || null;
        if (y_) this.l = y_.length;
    }

    /* svm_model Class */
    SVM.Model = function()
    {
        this.param = null;    // parameter
        this.nr_class = 2;        // number of classes, = 2 in regression/one class svm
        this.l = 0;            // total #SV
        this.SV = null;    // SVs (SV[l])
        this.sv_coef = null;    // coefficients for SVs in decision functions (sv_coef[k-1][l])
        this.rho = null;        // constants in decision functions (rho[k*(k-1)/2])
        this.probA = null;         // pariwise probability information
        this.probB = null;

        // for classification only

        this.label = null;        // label of each class (label[k])
        this.nSV = null;        // number of SVs for each class (nSV[k])
                    // nSV[0] + nSV[1] + ... + nSV[k-1] = l
    }

    /* Kernel Class*/
    SVM.Kernel = function(l, x_, param)
    {
        if(arguments.length==0){
            this.kernel_type = 0;
            this.degree = 0;
            this.gamma = 0.0;
            this.coef0 = 0.0;
            this.x = null;
            this.x_square = null;
        }
        else{
            this.kernel_type = param.kernel_type;
            this.degree = param.degree;
            this.gamma = param.gamma;
            this.coef0 = param.coef0;
            this.x = SVM.clone(x_);
            if(this.kernel_type == SVM.RBF){
                this.x_square = SVM.arr(l, 0.0);
                for(var i=0; i<l; i++)
                    this.x_square[i] = dot(x[i],x[i]);
            }
            else 
                this.x_square = null;
        }
    }
    
    function powi(base, times){
        var tmp = base, ret = 1.0;

        for(var t=times; t>0; t/=2)
        {
            if(t%2==1) ret*=tmp;
            tmp = tmp * tmp;
        }
        return ret;
    }
    
    function dot(x, y)
    {
        var sum = 0;
        var xlen = x.length;
        var ylen = y.length;
        var i = 0, j = 0;
        while(i < xlen && j < ylen)
        {
            //SVM.info('dot: '+i+' '+j+' ('+x[i]+' * '+y[j]+' = '+x[i] * y[j]);
            if(x[i].index == y[j].index)
                sum += x[i++].value * y[j++].value;
            else
            {
                if(x[i].index > y[j].index)
                    ++j;
                else
                    ++i;
            }
        }
        return sum;
    }
    
    function k_function(x, y, param)
    {
        switch(param.kernel_type)
        {
            case SVM.LINEAR: return dot(x,y);
            case SVM.POLY: return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
            case SVM.RBF:
            {
                var sum = 0;
                var xlen = x.length;
                var ylen = y.length;
                var i = 0, j = 0;
                
                while(i < xlen && j < ylen){
                    if(x[i].index == y[j].index){
                        var d = x[i++].value - y[j++].value;
                        sum += d*d;
                    }
                    else if(x[i].index > y[j].index){
                        sum += y[j].value * y[j].value;
                        ++j;
                    }
                    else{
                        sum += x[i].value * x[i].value;
                        ++i;
                    }
                }

                while(i < xlen){
                    sum += x[i].value * x[i].value;
                    ++i;
                }
                while(j < ylen){
                    sum += y[j].value * y[j].value;
                    ++j;
                }

                return Math.exp(-param.gamma*sum);
            }
            case SVM.SIGMOID: return tanh(param.gamma*dot(x,y)+param.coef0);
            case SVM.PRECOMPUTED: return    x[(int)(y[0].value)].value;
        }
    }

    SVM.Kernel.prototype = {
        get_Q: function(column, len){},
        get_QD: function(){},
        swap_index: function(i,j){
            _=this.x[i]; this.x[i]=this.x[j]; this.x[j]=_;
            if(this.x_square) {_=this.x_square[i]; this.x_square[i]=this.x_square[j]; this.x_square[j]=_;}
        },
        kernel_function: function(i, j){
            //SVM.info('x['+i+'] '+this.x[i]+' x['+j+'] '+this.x[j]+' dot '+dot(this.x[i], this.x[j]))
            switch(this.kernel_type)
            {
                case SVM.LINEAR: return dot(this.x[i], this.x[j]);
                case SVM.POLY: return powi(this.gamma * dot(this.x[i], this.x[j]) + this.coef0, this.degree);
                case SVM.RBF: return Math.exp(-this.gamma * (this.x_square[i] + this.x_square[j] - 2 * dot(this.x[i], this.x[j])));
                case SVM.SIGMOID: return tanh(this.gamma * dot(this.x[i], this.x[j]) + this.coef0);
                case SVM.PRECOMPUTED: return this.x[i][Math.floor(this.x[j][0].value)].value;
            }
        }
    };
    
    SVM.LOWER_BOUND = 0;
    SVM.UPPER_BOUND = 1;
    SVM.FREE = 2;
    SVM.INFINITY = 1/0;
    
    SVM.SolutionInfo = function()
    {
        this.obj = 0.0;
        this.rho = 0.0;
        this.upper_bound_p = 0.0;
        this.lower_bound_p = 0.0;
        this.r = 0.0;
    };

    /* Solver Class */
    // An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
    // Solves:
    //
    //    min 0.5(\alpha^T Q \alpha) + p^T \alpha
    //
    //        y^T \alpha = \delta
    //        y_i = +1 or -1
    //        0 <= alpha_i <= Cp for y_i = 1
    //        0 <= alpha_i <= Cn for y_i = -1
    //
    // Given:
    //
    //    Q, p, y, Cp, Cn, and an initial feasible point \alpha
    //    l is the size of vectors and matrices
    //    eps is the stopping tolerance
    //
    // solution will be put in \alpha, objective value will be put in obj
    //
    SVM.Solver = function()
    {
        this.active_size = 0;
        this.Cp = 0.0;
        this.Cn = 0.0;
        
        this.y = null;
        this.G = null;
        this.alpha_status = null;
        this.alpha = null;
        this.Q = null;
        this.QD = null;
        this.eps = 0.0;
        this.p = null;
        this.active_set = null;
        this.G_bar = null;
        this.l = 0;
        this.unshrink = false;
    }

    SVM.Solver.prototype = {
        is_upper_bound: function(i){
            return this.alpha_status[i] == SVM.UPPER_BOUND;
        },
        is_lower_bound: function(i){
            return this.alpha_status[i] == SVM.LOWER_BOUND;
        },
        is_free: function(i){
            return this.alpha_status[i] == SVM.FREE;
        },
        get_C: function(i){
            return (this.y[i]>0)?this.Cp:this.Cn;
        },
        update_alpha_status: function(i){
            if(this.alpha[i] >= this.get_C(i))
                this.alpha_status[i] = SVM.UPPER_BOUND;
            else if(this.alpha[i] <= 0)
                this.alpha_status[i] = SVM.LOWER_BOUND;
            else 
                this.alpha_status[i] = SVM.FREE;
        },
        swap_index: function(i,j){
            this.Q.swap_index(i,j);
            _=this.y[i]; this.y[i]=this.y[j]; this.y[j]=_;
            _=this.G[i]; this.G[i]=this.G[j]; this.G[j]=_;
            _=this.alpha_status[i]; this.alpha_status[i]=this.alpha_status[j]; this.alpha_status[j]=_;
            _=this.alpha[i]; this.alpha[i]=this.alpha[j]; this.alpha[j]=_;
            _=this.p[i]; this.p[i]=this.p[j]; this.p[j]=_;
            _=this.active_set[i]; this.active_set[i]=this.active_set[j]; this.active_set[j]=_;
            _=this.G_bar[i]; this.G_bar[i]=this.G_bar[j]; this.G_bar[j]=_;
        },
        reconstruct_gradient: function(){
            if(this.active_size==this.l) return;
            
            var i,j;
            var nr_free = 0;
            
            for(j=this.active_size;j<l;j++)
                this.G[j] = this.G_bar[j] + this.p[j];

            for(j=0;j<this.active_size;j++)
                if(this.is_free(j))
                    nr_free++;

            //if(2*nr_free < this.active_size) svm.info("\nWarning: using -h 0 may be faster\n");
            if (nr_free*l > 2*this.active_size*(this.l-this.active_size)) {
                for(i=this.active_size;i<this.l;i++) {
                    var Q_i = this.Q.get_Q(i,this.active_size);
                    for(j=0; j<this.active_size; j++)
                        if(this.is_free(j))
                            this.G[i] += this.alpha[j] * Q_i[j];
                }    
            }
            else {
                for(i=0; i<this.active_size; i++)
                    if(this.is_free(i)) {
                        var Q_i = this.Q.get_Q(i,this.l);
                        var alpha_i = this.alpha[i];
                        for(j=this.active_size; j<this.l; j++)
                            this.G[j] += alpha_i * Q_i[j];
                    }
            }
        },
        Solve:function(l, Q, p_, y_, alpha_, Cp, Cn, eps, si, shrinking)
        {
            SVM.info("Solve..")
            this.l = l;
            this.Q = Q;
            this.QD = Q.get_QD();
            this.p = p_.slice(0);
            this.y = y_.slice(0);
            this.alpha = alpha_.slice(0);
            this.Cp = Cp;
            this.Cn = Cn;
            this.eps = eps;
            this.unshrink = false;
            
            //...
            // initialize alpha_status
            this.alpha_status = SVM.arr(this.l, 0);
            for(var i=0; i<this.l; i++)
                this.update_alpha_status(i);
                
            // initialize active set (for shrinking)
            this.active_set = SVM.arr(this.l, 0);
            for(var i=0; i<this.l; i++)
                this.active_set[i] = i;
            this.active_size = this.l;
            
            // initialize gradient
            this.G = SVM.arr(this.l, 0.0);
            this.G_bar = SVM.arr(this.l, 0.0);
            for(var i=0; i<l; i++) {
                this.G[i] = this.p[i];
                this.G_bar[i] = 0;
            }
            for(var i=0; i<this.l; i++)
                if(!this.is_lower_bound(i)) {
                    var Q_i = this.Q.get_Q(i,this.l);
                    var alpha_i = this.alpha[i];
                    
                    for(var j=0; j<this.l; j++)
                        this.G[j] += alpha_i * Q_i[j];
                    if(this.is_upper_bound(i))
                        for(var j=0; j<this.l; j++)
                            this.G_bar[j] += this.get_C(i) * Q_i[j];
                }
                
            // optimization step
            var iter = 0;
            var counter = Math.min(l,1000)+1;
            var working_set = SVM.arr(2, 0);
                
            while(true) {
                // show progress and do shrinking
                if(--counter == 0) {
                    counter = Math.min(this.l, 1000);
                    if(shrinking!=0) this.do_shrinking();
                    SVM.info(".");
                }

                if(this.select_working_set(working_set) != 0) {
                    // reconstruct the whole gradient
                    this.reconstruct_gradient();
                    // reset active set size and check
                    this.active_size = l;
                    SVM.info("*");
                    if(this.select_working_set(working_set)!=0)
                        break;
                    else
                        counter = 1;    // do shrinking next iteration
                }
                
                var i = working_set[0];
                var j = working_set[1];
                ++iter;
                
                // update alpha[i] and alpha[j], handle bounds carefully
                var Q_i = this.Q.get_Q(i, this.active_size);
                var Q_j = this.Q.get_Q(j, this.active_size);

                var C_i = this.get_C(i);
                var C_j = this.get_C(j);

                var old_alpha_i = this.alpha[i];
                var old_alpha_j = this.alpha[j];

                if(this.y[i] != this.y[j]) {
                    var quad_coef = this.QD[i] + this.QD[j] + 2* Q_i[j];
                    if (quad_coef <= 0) quad_coef = 1e-12;
                    var delta = (-this.G[i] - this.G[j]) / quad_coef;
                    var diff = this.alpha[i] - this.alpha[j];
                    this.alpha[i] += delta;
                    this.alpha[j] += delta;
                
                    if(diff > 0) {
                        if(this.alpha[j] < 0) {
                            this.alpha[j] = 0;
                            this.alpha[i] = diff;
                        }
                    }
                    else {
                        if(this.alpha[i] < 0) {
                            this.alpha[i] = 0;
                            this.alpha[j] = -diff;
                        }
                    }
                    if(diff > C_i - C_j) {
                        if(this.alpha[i] > C_i) {
                            this.alpha[i] = C_i;
                            this.alpha[j] = C_i - diff;
                        }
                    }
                    else {
                        if(this.alpha[j] > C_j) {
                            this.alpha[j] = C_j;
                            this.alpha[i] = C_j + diff;
                        }
                    }
                }
                else {
                    quad_coef = this.QD[i] + this.QD[j] - 2 * Q_i[j];
                    if (quad_coef <= 0)
                        quad_coef = 1e-12;
                    var delta = (this.G[i] - this.G[j]) / quad_coef;
                    var sum = this.alpha[i] + this.alpha[j];
                    this.alpha[i] -= delta;
                    this.alpha[j] += delta;

                    if(sum > C_i) {
                        if(this.alpha[i] > C_i) {
                            this.alpha[i] = C_i;
                            this.alpha[j] = sum - C_i;
                        }
                    }
                    else {
                        if(this.alpha[j] < 0) {
                            this.alpha[j] = 0;
                            this.alpha[i] = sum;
                        }
                    }
                    if(sum > C_j) {
                        if(this.alpha[j] > C_j) {
                            this.alpha[j] = C_j;
                            this.alpha[i] = sum - C_j;
                        }
                    }
                    else {
                        if(this.alpha[i] < 0) {
                            this.alpha[i] = 0;
                            this.alpha[j] = sum;
                        }
                    }
                }

                // update G
                delta_alpha_i = this.alpha[i] - old_alpha_i;
                delta_alpha_j = this.alpha[j] - old_alpha_j;

                for(var k=0; k<this.active_size; k++) {
                    this.G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j;
                }

                // update alpha_status and G_bar
                var ui = this.is_upper_bound(i);
                var uj = this.is_upper_bound(j);
                this.update_alpha_status(i);
                this.update_alpha_status(j);
                
                if(ui != this.is_upper_bound(i)) {
                    Q_i = this.Q.get_Q(i, this.l);
                    if(ui)
                        for(var k=0; k<this.l; k++)
                            this.G_bar[k] -= C_i * Q_i[k];
                    else
                        for(var k=0; k<this.l; k++)
                            this.G_bar[k] += C_i * Q_i[k];
                }

                if(uj != this.is_upper_bound(j)) {
                    Q_j = this.Q.get_Q(j,l);
                    if(uj)
                        for(var k=0; k<this.l; k++)
                            this.G_bar[k] -= C_j * Q_j[k];
                    else
                        for(k=0; k<this.l; k++)
                            this.G_bar[k] += C_j * Q_j[k];
                }

            }
            
            // calculate rho
            si.rho = this.calculate_rho();

            // calculate objective value
            var v = 0;
            for(var i=0; i<this.l; i++)
                v += this.alpha[i] * (this.G[i] + this.p[i]);

            si.obj = v/2;
            
            // put back the solution
            for(var i=0; i<this.l; i++)
                alpha_[this.active_set[i]] = this.alpha[i];

            si.upper_bound_p = this.Cp;
            si.upper_bound_n = this.Cn;

            SVM.info("\noptimization finished, #iter = "+iter+"\n");
        },
        
        select_working_set: function(working_set){
            // return i,j such that
            // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
            // j: mimimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
            Gmax = -SVM.INFINITY;
            Gmax2 = -SVM.INFINITY;
            Gmax_idx = -1;
            Gmin_idx = -1;
            obj_diff_min = SVM.INFINITY;
            
            for(var t=0; t<this.active_size;t++) {
                SVM.info('y['+t+'] = '+this.y[t]+' G['+t+'] = '+this.G[t]);
                if(this.y[t] == +1){
                    if(!this.is_upper_bound(t))
                        if(-this.G[t] >= Gmax) {
                            Gmax = -this.G[t];
                            Gmax_idx = t;
                        }
                }
                else {
                    if(!this.is_lower_bound(t))
                        if(this.G[t] >= Gmax) {
                            Gmax = this.G[t];
                            Gmax_idx = t;
                        }
                }
            }
            
            i = Gmax_idx;
            Q_i = null;
            // null Q_i not accessed: Gmax=-INF if i=-1
            if(i != -1) Q_i = this.Q.get_Q(i, this.active_size);
            alert('Gmax  '+Gmax+' Gmax2  '+Gmax2+' '+this.eps);
            for(var j=0; j<this.active_size; j++) {
                if(this.y[j] == +1) {
                    if (!this.is_lower_bound(j)){
                        grad_diff = Gmax + this.G[j];
                        if (this.G[j] >= Gmax2) Gmax2 = this.G[j];
                        
                        if (grad_diff > 0) {
                            var obj_diff; 
                            var quad_coef = this.QD[i] + this.QD[j] - 2.0 * this.y[i] * Q_i[j];
                            
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = -(grad_diff * grad_diff) / 1e-12;
        
                            if (obj_diff <= obj_diff_min) {
                                Gmin_idx = j;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
                else {
                    if (!this.is_upper_bound(j)) {
                        grad_diff = Gmax - this.G[j];
                        if (-this.G[j] >= Gmax2) Gmax2 = -this.G[j];
                        
                        if (grad_diff > 0) {
                            var obj_diff; 
                            var quad_coef = this.QD[i] + this.QD[j] + 2.0 * this.y[i] * Q_i[j];
                            
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = -(grad_diff * grad_diff) / 1e-12;
        
                            if (obj_diff <= obj_diff_min) {
                                Gmin_idx = j;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
            }
            
            if(Gmax + Gmax2 < this.eps)
                return 1;

            working_set[0] = Gmax_idx;
            working_set[1] = Gmin_idx;
            return 0;
        },
        
        be_shrunk: function(i, Gmax1, Gmax2){
            if(this.is_upper_bound(i)) {
                if(this.y[i] == +1)
                    return (-this.G[i] > Gmax1);
                else
                    return (-this.G[i] > Gmax2);
            }
            else if(this.is_lower_bound(i)) {
                if(this.y[i] == +1)
                    return (this.G[i] > Gmax2);
                else    
                    return (this.G[i] > Gmax1);
            }
            else
                return false;
        },
        
        do_shrinking: function()
        {
            var Gmax1 = -INF;        // max { -y_i * grad(f)_i | i in I_up(\alpha) }
            var Gmax2 = -INF;        // max { y_i * grad(f)_i | i in I_low(\alpha) }

            // find maximal violating pair first
            for(var i=0; i<this.active_size; i++) {
                if(this.y[i] == +1) {
                    if(!this.is_upper_bound(i)) {
                        if(-this.G[i] >= Gmax1)
                            Gmax1 = -this.G[i];
                    }
                    if(!this.is_lower_bound(i)) {
                        if(this.G[i] >= Gmax2)
                            Gmax2 = this.G[i];
                    }
                }
                else {
                    if(!this.is_upper_bound(i)) {
                        if(-this.G[i] >= Gmax2)
                            Gmax2 = -this.G[i];
                    }
                    if(!this.is_lower_bound(i)) {
                        if(this.G[i] >= Gmax1)
                            Gmax1 = this.G[i];
                    }
                }
            }

            if(this.unshrink == false && Gmax1 + Gmax2 <= this.eps*10) {
                this.unshrink = true;
                this.reconstruct_gradient();
                this.active_size = l;
            }

            for(var i=0; i<this.active_size; i++)
                if (this.be_shrunk(i, Gmax1, Gmax2)) {
                    this.active_size--;
                    while (this.active_size > i) {
                        if (!this.be_shrunk(this.active_size, Gmax1, Gmax2)) {
                            this.swap_index(i, this.active_size);
                            break;
                        }
                        this.active_size--;
                    }
                }
        },
        
        calculate_rho: function() {
            var r, nr_free = 0;
            var ub = SVM.INFINITY, lb = -SVM.INFINITY, sum_free = 0;
            for(var i=0; i<this.active_size; i++) {
                var yG = this.y[i] * this.G[i];

                if(this.is_lower_bound(i)) {
                    if(this.y[i] > 0)
                        ub = Math.min(ub,yG);
                    else
                        lb = Math.max(lb,yG);
                }
                else if(this.is_upper_bound(i)) {
                    if(this.y[i] < 0)
                        ub = Math.min(ub,yG);
                    else
                        lb = Math.max(lb,yG);
                }
                else {
                    ++nr_free;
                    sum_free += yG;
                }
            }

            if(nr_free>0)
                r = sum_free/nr_free;
            else
                r = (ub+lb)/2;

            return r;
        }
        
    };
    
    //
    // Solver for nu-svm classification and regression
    //
    // additional constraint: e^T \alpha = constant
    //
    SVM.Solver_NU = function(){
        SVM.Solver.call(this);
        
        this.si = new SVM.SolutionInfo();
    }
    
    SVM.Solver_NU.prototype = new SVM.Solver();
    SVM.Solver_NU.prototype.constructor = SVM.Solver_NU;
    SVM.Solver_NU.prototype.super = SVM.Solver;
    
    SVM.Solver_NU.prototype.Solve = function(l, Q, p_, y_, alpha_, Cp, Cn, eps, si, shrinking){
        this.si = si;
        this.super.Solve(l, Q, p_, y_, alpha_, Cp, Cn, eps, si, shrinking);
    }
    
    // return 1 if already optimal, return 0 otherwise
    SVM.Solver_NU.prototype.select_working_set = function(working_set)
    {
        // return i,j such that y_i = y_j and
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
    
        var Gmaxp = -SVM.INFINITY;
        var Gmaxp2 = -SVM.INFINITY;
        var Gmaxp_idx = -1;
    
        var Gmaxn = -SVM.INFINITY;
        var Gmaxn2 = -SVM.INFINITY;
        var Gmaxn_idx = -1;
    
        var Gmin_idx = -1;
        var obj_diff_min = SVM.INFINITY;
    
        for(var t=0; t<this.active_size; t++)
            if(this.y[t] == +1) {
                if(!this.is_upper_bound(t))
                    if(-this.G[t] >= Gmaxp) {
                        Gmaxp = -this.G[t];
                        Gmaxp_idx = t;
                    }
            }
            else {
                if(!this.is_lower_bound(t))
                    if(this.G[t] >= Gmaxn) {
                        Gmaxn = this.G[t];
                        Gmaxn_idx = t;
                    }
            }
    
        var ip = Gmaxp_idx;
        var in_ = Gmaxn_idx;
        var  Q_ip = null;
        var  Q_in = null;
        if(ip != -1) // null Q_ip not accessed: Gmaxp=-INF if ip=-1
            Q_ip = this.Q.get_Q(ip,active_size);
        if(in_ != -1)
            Q_in = this.Q.get_Q(in_,active_size);
    
        for(var j=0; j<this.active_size; j++) {
            if(this.y[j] == +1) {
                if (!this.is_lower_bound(j))    {
                    var grad_diff = Gmaxp + this.G[j];
                    if (this.G[j] >= Gmaxp2) Gmaxp2 = this.G[j];
                    if (grad_diff > 0) {
                        var obj_diff; 
                        var quad_coef = this.QD[ip] + this.QD[j] - 2 * Q_ip[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;
    
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
            else {
                if (!this.is_upper_bound(j)) {
                    var grad_diff = Gmaxn - this.G[j];
                    if (-this.G[j] >= Gmaxn2)
                        Gmaxn2 = -this.G[j];
                    if (grad_diff > 0) {
                        var obj_diff; 
                        var quad_coef = this.QD[in_] + this.QD[j] - 2 * Q_in[j];
                        if (quad_coef > 0)
                            obj_diff = -(grad_diff * grad_diff) / quad_coef;
                        else
                            obj_diff = -(grad_diff * grad_diff) / 1e-12;
    
                        if (obj_diff <= obj_diff_min) {
                            Gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if(Math.max(Gmaxp + Gmaxp2,Gmaxn + Gmaxn2) < this.eps)
            return 1;
    
        if(this.y[Gmin_idx] == +1)
            working_set[0] = Gmaxp_idx;
        else
            working_set[0] = Gmaxn_idx;
        working_set[1] = Gmin_idx;
    
        return 0;
    }
    
    SVM.Solver_NU.prototype.be_shrunk = function(i, Gmax1, Gmax2, Gmax3, Gmax4)
    {
        if(this.is_upper_bound(i)){
            if(this.y[i] == +1)
                return(-this.G[i] > Gmax1);
            else    
                return(-this.G[i] > Gmax4);
        }
        else if(this.is_lower_bound(i)){
            if(this.y[i] == +1)
                return(this.G[i] > Gmax2);
            else    
                return(this.G[i] > Gmax3);
        }
        else
            return(false);
    }
    
    SVM.Solver_NU.prototype.do_shrinking = function()
    {
        var Gmax1 = -SVM.INFINITY;    // max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
        var Gmax2 = -SVM.INFINITY;    // max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
        var Gmax3 = -SVM.INFINITY;    // max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
        var Gmax4 = -SVM.INFINITY;    // max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }
 
        // find maximal violating pair first
        for(var i=0; i<this.active_size; i++) {
            if(!this.is_upper_bound(i)){
                if(this.y[i] == +1){
                    if(-this.G[i] > Gmax1) Gmax1 = -this.G[i];
                }
                else
                    if(-this.G[i] > Gmax4) Gmax4 = -this.G[i];
            }
            if(!this.is_lower_bound(i)){
                if(this.y[i]==+1){    
                    if(this.G[i] > Gmax2) Gmax2 = this.G[i];
                }
                else
                    if(this.G[i] > Gmax3) Gmax3 = this.G[i];
            }
        }

        if(this.unshrink == false && Math.max(Gmax1+Gmax2,Gmax3+Gmax4) <= this.eps*10) {
            this.unshrink = true;
            this.reconstruct_gradient();
            this.active_size = l;
        }

        for(var i=0; i<this.active_size; i++)
            if (this.be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)) {
                this.active_size--;
                while (this.active_size > i) {
                    if (!this.be_shrunk(this.active_size, Gmax1, Gmax2, Gmax3, Gmax4)) {
                        this.swap_index(i, this.active_size);
                        break;
                    }
                    this.active_size--;
                }
            }
    }
    
    SVM.Solver_NU.prototype.calculate_rho = function()
    {
        var nr_free1 = 0,nr_free2 = 0;
        var ub1 = INF, ub2 = INF;
        var lb1 = -INF, lb2 = -INF;
        var sum_free1 = 0, sum_free2 = 0;

        for(var i=0; i<this.active_size; i++){
            if(this.y[i] == +1){
                if(this.is_lower_bound(i))
                    ub1 = Math.min(ub1, this.G[i]);
                else if(this.is_upper_bound(i))
                    lb1 = Math.max(lb1, this.G[i]);
                else{
                    ++nr_free1;
                    sum_free1 += this.G[i];
                }
            }
            else {
                if(this.is_lower_bound(i))
                    ub2 = Math.min(ub2, this.G[i]);
                else if(this.is_upper_bound(i))
                    lb2 = Math.max(lb2, this.G[i]);
                else{
                    ++nr_free2;
                    sum_free2 += G[i];
                }
            }
        }

        var r1,r2;
        if(nr_free1 > 0)
            r1 = sum_free1/nr_free1;
        else
            r1 = (ub1+lb1)/2;

        if(nr_free2 > 0)
            r2 = sum_free2/nr_free2;
        else
            r2 = (ub2+lb2)/2;

        si.r = (r1+r2)/2;
        return (r1-r2)/2;
    }
    
    //SVC_Q Kernel
    SVM.SVC_Q = function(prob, param, y_)
    {
        SVM.Kernel.call(this);
        this.super(prob.l, prob.x, param);
        
        this.y = SVM.clone(y_);
        this.cache = new SVM.Cache(prob.l, (param.cache_size*(1<<20)));
        this.QD = SVM.arr(prob.l, 0.0);
        for(var i=0; i<prob.l; ++i){
            this.QD[i] = this.kernel_function(i,i)
            SVM.info("QD_i "+this.QD[i]);
        }
    }
    
    SVM.SVC_Q.prototype = new SVM.Kernel();
    SVM.SVC_Q.prototype.constructor = SVM.SVC_Q;
    SVM.SVC_Q.prototype.super = SVM.Kernel;
    
    SVM.SVC_Q.prototype.get_Q = function(i, len)
    {
        data = SVM.arr(1, Array);
        var start, j;
        if((start = this.cache.get_data(i,data,len)) < len){
            for(j=start; j<len; j++)
                data[0][j] = (this.y[i] * this.y[j] * this.kernel_function(i,j));
        }
        return data[0];
    }
    
    SVM.SVC_Q.prototype.get_QD = function(){ return this.QD; }
    
    SVM.SVC_Q.prototype.swap_index = function(i, j)
    {
        this.cache.swap_index(i,j);
        this.super.swap_index(i,j);
        _=this.y[i]; this.y[i]=this.y[j]; this.y[j]=_;
        _=this.QD[i]; this.QD[i]=this.QD[j]; this.QD[j]=_;
    }

    //ONECLASS_Q Kernel
    SVM.ONECLASS_Q = function(prob, param)
    {
        SVM.Kernel.call(this);
        this.super(prob.l, prob.x, param);
        
        this.cache = new SVM.Cache(prob.l, param.cache_size*(1<<20));
        this.QD = SVM.arr(prob.l, 0.0);
        for(var i=0; i<prob.l; ++i)
            this.QD[i] = this.kernel_function(i,i);
    }
    
    SVM.ONECLASS_Q.prototype = new SVM.Kernel();
    SVM.ONECLASS_Q.prototype.constructor = SVM.ONECLASS_Q;
    SVM.ONECLASS_Q.prototype.super = SVM.Kernel;
    
    SVM.ONECLASS_Q.prototype.get_Q = function(i, len)
    {
        var data = SVM.arr(1, Array);
        var start, j;
        if((start = this.cache.get_data(i,data,len)) < len){
            for(j=start; j<len; j++)
                data[0][j] = this.kernel_function(i,j);
        }
        return data[0];
    }
    
    SVM.ONECLASS_Q.prototype.get_QD = function()
    {
        return this.QD;
    }
    
    SVM.ONECLASS_Q.prototype.swap_index = function(i, j)
    {
        this.cache.swap_index(i,j);
        this.super.swap_index(i,j);
        _=QD[i]; QD[i]=QD[j]; QD[j]=_;
    }
    
    //SVR_Q Kernel
    SVM.SVR_Q = function(prob, param)
    {
        SVM.Kernel.call(this)
        this.super(prob.l, prob.x, param);
        
        this.l = prob.l;
        this.cache = new SVM.Cache(l, param.cache_size*(1<<20));
        this.QD = SVM.arr(2*this.l, 0.0);
        this.sign = SVM.arr(2*this.l, 0);
        this.index = SVM.arr(2*this.l, 0);
        for(var k=0; k<this.l; ++k){
            this.sign[k] = 1;
            this.sign[k+l] = -1;
            this.index[k] = k;
            this.index[k+l] = k;
            this.QD[k] = this.kernel_function(k,k);
            this.QD[k+l] = this.QD[k];
        }
        this.buffer = SVM.arr([2, 2*prob.l], 0.0);
        this.next_buffer = 0;
    }
    
    SVM.SVR_Q.prototype = new SVM.Kernel();
    SVM.SVR_Q.prototype.constructor = SVM.SVR_Q;
    SVM.SVR_Q.prototype.super = SVM.Kernel;
    
    SVM.SVR_Q.prototype.swap_index = function(i, j)
    {
        _=sign[i]; sign[i]=sign[j]; sign[j]=_;
        _=index[i]; index[i]=index[j]; index[j]=_;
        _=QD[i]; QD[i]=QD[j]; QD[j]=_;
    }
    
    SVM.SVR_Q.prototype.get_Q = function(i, len)
    {
        var data = SVM.arr(1, Array);
        var j, real_i = this.index[i];
        if(this.cache.get_data(real_i, data, this.l) < this.l)
        {
            for(j=0; j<this.l; j++)
                data[0][j] = this.kernel_function(real_i,j);
        }

        // reorder and copy
        var buf = SVM.clone(this.buffer[this.next_buffer]);
        this.next_buffer = 1 - this.next_buffer;
        var si = this.sign[i];
        for(j=0; j<len; j++)
            buf[j] = si * sign[j] * data[0][this.index[j]];
        return buf;
    }
    
    SVM.SVR_Q.prototype.get_QD = function(){ return this.QD; }
    
    //main svm class
    
    SVM.rand = Math.random;
    
    SVM.info = function(msg)
    {
        alert(msg);
    }
    
    SVM.solve_c_svc = function(prob, param, alpha, si, Cp, Cn)
    {
        var l = prob.l;
        var minus_ones = SVM.arr(l, 0.0);
        
        var y = SVM.arr(l, 0);
        SVM.info("solve_c_svc");
        var i;
        for(i=0; i<l; i++){
            alpha[i] = 0;
            minus_ones[i] = -1;
            if(prob.y[i] > 0) y[i] = +1; else y[i] = -1;
        }

        var s = new SVM.Solver();
        s.Solve(l, new SVM.SVC_Q(prob,param,y), minus_ones, y,
            alpha, Cp, Cn, param.eps, si, param.shrinking);

        var sum_alpha=0;
        for(i=0; i<l; i++)
            sum_alpha += alpha[i];
            
        SVM.info('Cp = '+Cp+' Cn = '+Cn)
        if (Cp == Cn)
            SVM.info("nu = "+sum_alpha/(Cp*prob.l)+"\n");

        for(i=0; i<l; i++)
            alpha[i] *= y[i];
    }
    
    SVM.solve_nu_svc = function(prob, param, alpha, si)
    {
        var i;
        var l = prob.l;
        var nu = param.nu;

        var y = SVM.arr(l, 0);

        for(i=0; i<l; i++)
            if(prob.y[i]>0)
                y[i] = +1;
            else
                y[i] = -1;

        var sum_pos = nu*l/2;
        var sum_neg = nu*l/2;

        for(i=0;i<l;i++)
            if(y[i] == +1){
                alpha[i] = Math.min(1.0,sum_pos);
                sum_pos -= alpha[i];
            }
            else{
                alpha[i] = Math.min(1.0,sum_neg);
                sum_neg -= alpha[i];
            }

        var zeros = SVM.arr(l, 0.0);

        var s = new SVM.Solver_NU();
        s.Solve(l, new SVM.SVC_Q(prob,param,y), zeros, y,
            alpha, 1.0, 1.0, param.eps, si, param.shrinking);
        var r = si.r;

        SVM.info("C = "+1/r+"\n");

        for(i=0; i<l; i++)
            alpha[i] *= y[i]/r;

        si.rho /= r;
        si.obj /= (r*r);
        si.upper_bound_p = 1/r;
        si.upper_bound_n = 1/r;
    }
    
    SVM.solve_one_class = function(prob, param, alpha, si)
    {
        var l = prob.l;
        var zeros = SVM.arr(l, 0.0);
        var ones = SVM.arr(l, 0);
        var i;

        var n = parseInt(param.nu * prob.l);    // # of alpha's at upper bound

        for(i=0; i<n; i++)
            alpha[i] = 1;
            
        if(n < prob.l)
            alpha[n] = param.nu * prob.l - n;
            
        for(i=n+1; i<l; i++)
            alpha[i] = 0;

        for(i=0; i<l; i++){
            zeros[i] = 0;
            ones[i] = 1;
        }

        s = new SVM.Solver();
        s.Solve(l, new ONE_CLASS_Q(prob,param), zeros, ones,
            alpha, 1.0, 1.0, param.eps, si, param.shrinking);
    }
    
    SVM.solve_epsilon_svr = function(prob, param, alpha, si)
    {
        var l = prob.l;
        var alpha2 = SVM.arr(2*l, 0.0);
        var linear_term = SVM.arr(2*l, 0.0);
        var y = SVM.arr(2*l, 0);
        var i;

        for(i=0; i<l; i++){
            alpha2[i] = 0;
            linear_term[i] = param.p - prob.y[i];
            y[i] = 1;

            alpha2[i+l] = 0;
            linear_term[i+l] = param.p + prob.y[i];
            y[i+l] = -1;
        }

        s = new SVM.Solver();
        s.Solve(2*l, new SVM.SVR_Q(prob,param), linear_term, y,
            alpha2, param.C, param.C, param.eps, si, param.shrinking);

        var sum_alpha = 0;
        for(i=0; i<l; i++){
            alpha[i] = alpha2[i] - alpha2[i+l];
            sum_alpha += Math.abs(alpha[i]);
        }
        SVM.info("nu = "+sum_alpha/(param.C*l)+"\n");
    }
    
    SVM.solve_nu_svr = function(prob, param, alpha, si)
    {
        var l = prob.l;
        var C = param.C;
        var alpha2 = SVM.arr(2*l, 0.0);
        var linear_term = SVM.arr(2*l, 0.0);
        var y = SVM.arr(2*l, 0);
        var i;

        var sum = C * param.nu * l / 2;
        for(i=0; i<l; i++){
            alpha2[i] = alpha2[i+l] = Math.min(sum,C);
            sum -= alpha2[i];
            
            linear_term[i] = - prob.y[i];
            y[i] = 1;

            linear_term[i+l] = prob.y[i];
            y[i+l] = -1;
        }

        s = new SVM.Solver_NU();
        s.Solve(2*l, new SVM.SVR_Q(prob,param), linear_term, y,
            alpha2, C, C, param.eps, si, param.shrinking);

        SVM.info("epsilon = "+(-si.r)+"\n");
        
        for(i=0; i<l; i++)
            alpha[i] = alpha2[i] - alpha2[i+l];
    }
    
    //
    // decision_function
    //
    SVM.decision_function = function()
    {
        this.alpha = null;
        this.rho = 0.0;
    }
    
    SVM.train_one = function(prob, param, Cp, Cn)
    {
        SVM.info("train_one");
        var alpha = SVM.arr(prob.l, 0.0);
        si = new SVM.SolutionInfo();
        switch(param.svm_type){
            case SVM.C_SVC:
                SVM.solve_c_svc(prob,param,alpha,si,Cp,Cn);
                break;
            case SVM.NU_SVC:
                SVM.solve_nu_svc(prob,param,alpha,si);
                break;
            case SVM.ONE_CLASS:
                SVM.solve_one_class(prob,param,alpha,si);
                break;
            case SVM.EPSILON_SVR:
                SVM.solve_epsilon_svr(prob,param,alpha,si);
                break;
            case SVM.NU_SVR:
                SVM.solve_nu_svr(prob,param,alpha,si);
                break;
        }

        SVM.info("obj = "+si.obj+", rho = "+si.rho+"\n");

        // output SVs
        var nSV = 0;
        var nBSV = 0;
        for(var i=0;i<prob.l;i++){
            if(Math.abs(alpha[i]) > 0){
                ++nSV;
                if(prob.y[i] > 0){
                    if(Math.abs(alpha[i]) >= si.upper_bound_p)
                    ++nBSV;
                }
                else{
                    if(Math.abs(alpha[i]) >= si.upper_bound_n)
                        ++nBSV;
                }
            }
        }

        SVM.info("nSV = "+nSV+", nBSV = "+nBSV+"\n");

        var f = new SVM.decision_function();
        f.alpha = alpha;
        f.rho = si.rho;
        return f;
    }
    
    // Platt's binary SVM Probablistic Output: an improvement from Lin et al.
    SVM.sigmoid_train = function(l, dec_values, labels, probAB)
    {
        var A, B;
        var prior1=0, prior0 = 0;
        var i;

        for (i=0; i<l; i++)
            if (labels[i] > 0) prior1+=1;
            else prior0+=1;
    
        var max_iter=100;    // Maximal number of iterations
        var min_step=1e-10;    // Minimal step taken in line search
        var sigma=1e-12;    // For numerically strict PD of Hessian
        var eps=1e-5;
        var hiTarget=(prior1+1.0)/(prior1+2.0);
        var loTarget=1/(prior0+2.0);
        var t= SVM.arr(l, 0.0);
        var fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
        var newA,newB,newf,d1,d2;
        var iter; 
    
        // Initial Point and Initial Fun Value
        A=0.0; B=Math.log((prior0+1.0)/(prior1+1.0));
        var fval = 0.0;

        for (i=0; i<l; i++){
            if (labels[i] > 0) 
                t[i] = hiTarget;
            else 
                t[i] = loTarget;
            fApB = dec_values[i]*A+B;
            if (fApB>=0)
                fval += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
            else
                fval += (t[i] - 1)*fApB +Math.log(1+Math.exp(fApB));
        }
        for (iter=0; iter<max_iter; iter++){
            // Update Gradient and Hessian (use H' = H + sigma I)
            h11 = sigma; // numerically ensures strict PD
            h22 = sigma;
            h21 = 0.0;
            g1 = 0.0;
            g2 = 0.0;
            for (i=0; i<l; i++){
                fApB = dec_values[i]*A+B;
                if (fApB >= 0){
                    p = Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
                    q = 1.0 / (1.0 + Math.exp(-fApB));
                }
                else{
                    p = 1.0 / (1.0 + Math.exp(fApB));
                    q = Math.exp(fApB) / (1.0 + Math.exp(fApB));
                }
                d2 = p * q;
                h11 += dec_values[i] * dec_values[i] * d2;
                h22 += d2;
                h21 += dec_values[i] * d2;
                d1 = t[i] - p;
                g1 += dec_values[i] * d1;
                g2 += d1;
            }

            // Stopping Criteria
            if (Math.abs(g1) < eps && Math.abs(g2) < eps) break;
            
            // Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21;
            dA =- (h22 * g1 - h21 * g2) / det;
            dB =- (-h21 * g1+ h11 * g2) / det;
            gd = g1 * dA + g2 * dB;

            stepsize = 1;        // Line Search
            while (stepsize >= min_step){
                newA = A + stepsize * dA;
                newB = B + stepsize * dB;

                // New function value
                newf = 0.0;
                for (i=0; i<l; i++){
                    fApB = dec_values[i] * newA + newB;
                    if (fApB >= 0)
                        newf += t[i] * fApB + Math.log(1 + Math.exp(-fApB));
                    else
                        newf += (t[i] - 1) * fApB + Math.log(1 + Math.exp(fApB));
                }
                // Check sufficient decrease
                if (newf < fval + 0.0001 * stepsize * gd){
                    A = newA;
                    B = newB;
                    fval = newf;
                    break;
                }
                else
                    stepsize = stepsize / 2.0;
            }
            
            if (stepsize < min_step){
                SVM.info("Line search fails in two-class probability estimates\n");
                break;
            }
        }
        
        if (iter >= max_iter)
            SVM.info("Reaching maximal iterations in two-class probability estimates\n");
        probAB[0] = A;
        probAB[1] = B;
    }
    
    SVM.sigmoid_predict = function(decision_value, A, B)
    {
        var fApB = decision_value*A+B;
        if (fApB >= 0)
            return Math.exp(-fApB) / (1.0 + Math.exp(-fApB));
        else
            return 1.0 / (1 + Math.exp(fApB)) ;
    }
    
    // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
    SVM.multiclass_probability = function(k, r, p)
    {
        var t,j;
        var iter = 0, max_iter=Math.max(100,k);
        var Q = SVM.arr([k, k], 0.0);
        var Qp = SVM.arr(k, 0.0);
        var pQp, eps = 0.005/k;
    
        for (t=0;t<k;t++){
            p[t] = 1.0 / k;  // Valid if k = 1
            Q[t][t] = 0;
            for (j=0; j<t; j++){
                Q[t][t] += r[j][t] * r[j][t];
                Q[t][j] = Q[j][t];
            }
            for (j=t+1; j<k; j++){
                Q[t][t] += r[j][t] * r[j][t];
                Q[t][j] = -r[j][t] * r[t][j];
            }
        }
        for (iter=0; iter<max_iter; iter++){
            // stopping condition, recalculate QP,pQP for numerical accuracy
            pQp = 0;
            for (t=0; t<k; t++){
                Qp[t] = 0;
                for (j=0; j<k; j++)
                    Qp[t] += Q[t][j] * p[j];
                pQp += p[t] * Qp[t];
            }
            var max_error=0;
            for (t=0; t<k; t++){
                var error = Math.abs(Qp[t] - pQp);
                if (error > max_error)
                    max_error = error;
            }
            if (max_error<eps) break;
        
            for (t=0; t<k; t++){
                var diff = (-Qp[t] + pQp) / Q[t][t];
                p[t] += diff;
                pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
                for (j=0; j<k; j++){
                    Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                    p[j] /= (1+diff);
                }
            }
        }
        if (iter >= max_iter)
            SVM.info("Exceeds max_iter in multiclass_prob\n");
    }
    
    SVM.binary_svc_probability = function(prob, param, Cp, Cn, probAB)
    {
        var i;
        var nr_fold = 5;
        var perm = SVM.arr(prob.l, 0);
        var dec_values = SVM.arr(prob.l, 0.0);

        // random shuffle
        for(i=0; i<prob.l; i++) perm[i] = i;
        for(i=0; i<prob.l; i++){
            var j = i+rand.nextInt(prob.l-i);
            _ = perm[i]; perm[i] = perm[j]; perm[j] = _;
        }
        for(i=0; i<nr_fold; i++){
            var begin = i * prob.l / nr_fold;
            var end = (i+1)*prob.l / nr_fold;
            var j,k;
            subprob = new SVM.Problem();

            subprob.l = prob.l - (end - begin);
            subprob.x = SVM.arr([subprob.l,subprob.l], SVM.Node);
            subprob.y = SVM.arr(subprob.l, 0.0);
            
            k=0;
            for(j=0;j<begin;j++)
            {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for(j=end;j<prob.l;j++)
            {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            var p_count=0,n_count=0;
            for(j=0;j<k;j++)
                if(subprob.y[j]>0)
                    p_count++;
                else
                    n_count++;
            
            if(p_count==0 && n_count==0)
                for(j=begin;j<end;j++)
                    dec_values[perm[j]] = 0;
            else if(p_count > 0 && n_count == 0)
                for(j=begin;j<end;j++)
                    dec_values[perm[j]] = 1;
            else if(p_count == 0 && n_count > 0)
                for(j=begin; j<end; j++)
                    dec_values[perm[j]] = -1;
            else{
                subparam = param.clone();
                subparam.probability = 0;
                subparam.C = 1.0;
                subparam.nr_weight = 2;
                subparam.weight_label = SVM.arr(2, 0);
                subparam.weight = SVM.arr(2, 0.0);
                subparam.weight_label[0] = +1;
                subparam.weight_label[1] = -1;
                subparam.weight[0] = Cp;
                subparam.weight[1] = Cn;
                submodel = SVM.train(subprob, subparam);
                for(j=begin; j<end; j++)
                {
                    var dec_value= SVM.arr(1, 0.0);
                    svm_predict_values(submodel, prob.x[perm[j]], dec_value);
                    dec_values[perm[j]]=dec_value[0];
                    // ensure +1 -1 order; reason not using CV subroutine
                    dec_values[perm[j]] *= submodel.label[0];
                }        
            }
        }        
        SVM.sigmoid_train(prob.l,dec_values,prob.y,probAB);
    }
    
    // Return parameter of a Laplace distribution 
    SVM.svr_probability = function(prob, param)
    {
        var i;
        var nr_fold = 5;
        var ymv = SVM.arr(prob.l, 0.0);
        var mae = 0;

        newparam = SVM.clone(param);
        newparam.probability = 0;
        SVM.cross_validation(prob,newparam,nr_fold,ymv);
        for(i=0; i<prob.l; i++)
        {
            ymv[i] = prob.y[i]-ymv[i];
            mae += Math.abs(ymv[i]);
        }        
        mae /= prob.l;
        var std = Math.sqrt(2*mae*mae);
        var count=0;
        mae = 0;
        for(i=0; i<prob.l; i++)
            if (Math.abs(ymv[i]) > 5*std) 
                count=count+1;
            else 
                mae+=Math.abs(ymv[i]);
        mae /= (prob.l-count);
        SVM.info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="+mae+"\n");
        return mae;
    }
    
    // label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
    // perm, length l, must be allocated before calling this subroutine
    SVM.group_classes = function(prob, nr_class_ret, label_ret, start_ret, count_ret, perm)
    {
        var l = prob.l;
        var max_nr_class = 16;
        var nr_class = 0;
        var label = SVM.arr(max_nr_class, 0);
        var count = SVM.arr(max_nr_class, 0);
        var data_label = SVM.arr(l, 0);
        var i;

        for(i=0; i<l; i++){
            var this_label = Math.floor(prob.y[i]);
            var j;
            for(j=0; j<nr_class; j++){
                if(this_label == label[j]){
                    ++count[j];
                    break;
                }
            }
            data_label[i] = j;
            if(j == nr_class){
                if(nr_class == max_nr_class){
                    max_nr_class *= 2;
                    var new_data = SVM.arr(max_nr_class, 0);
                    System.arraycopy(label,0,new_data,0,label.length);
                    label = new_data;
                    new_data = SVM.arr(max_nr_class, 0);
                    System.arraycopy(count,0,new_data,0,count.length);
                    count = new_data;
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        var start = SVM.arr(nr_class, 0);
        start[0] = 0;
        for(i=1; i<nr_class; i++)
            start[i] = start[i-1] + count[i-1];
        for(i=0; i<l; i++){
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        start[0] = 0;
        for(i=1; i<nr_class; i++)
            start[i] = start[i-1] + count[i-1];

        nr_class_ret[0] = nr_class;
        label_ret[0] = label;
        start_ret[0] = start;
        count_ret[0] = count;
    }
    
    SVM.check_probability_model = function(model)
    {
        if (((model.param.svm_type == SVM.C_SVC || model.param.svm_type == SVM.NU_SVC) && model.probA != null && model.probB != null)||
        ((model.param.svm_type == SVM.EPSILON_SVR || model.param.svm_type == SVM.NU_SVR) && model.probA!=null))
            return 1;
        else
            return 0;
    }
    
    //
    // Interface functions
    //
    SVM.train = function(prob, param)
    {
        model = new SVM.Model();
        model.param = param;

        if(param.svm_type == SVM.ONE_CLASS || param.svm_type == SVM.EPSILON_SVR || param.svm_type == SVM.NU_SVR)
        {
            // regression or one-class-svm
            model.nr_class = 2;
            model.label = null;
            model.nSV = null;
            model.probA = null; model.probB = null;
            //model.sv_coef = new double[1][];
            model.sv_coef = SVM.arr(1, Array);
            
            if(param.probability == 1 && (param.svm_type == SVM.EPSILON_SVR || param.svm_type == SVM.NU_SVR)){
                model.probA = [0.0];
                model.probA[0] = svm_svr_probability(prob,param);
            }

            f = SVM.train_one(prob,param,0,0);
            model.rho = [0.0];
            model.rho[0] = f.rho;

            var nSV = 0;
            var i;
            for(i=0;i<prob.l;i++)
                if(Math.abs(f.alpha[i]) > 0) ++nSV;
            model.l = nSV;
            //model.SV = new svm_node[nSV][];
            model.SV = SVM.arr(nSV, Array);

            model.sv_coef[0] = SVM.arr(nSV, 0.0);
            var j = 0;
            for(i=0;i<prob.l;i++)
                if(Math.abs(f.alpha[i]) > 0){
                    model.SV[j] = prob.x[i];
                    model.sv_coef[0][j] = f.alpha[i];
                    ++j;
                }
        }
        else{
            // classification
            var l = prob.l;
            var tmp_nr_class = SVM.arr(1, 0);
            //int[][] tmp_label = new int[1][];
            var tmp_label = SVM.arr(1, Array);
            //int[][] tmp_start = new int[1][];
            var tmp_start = SVM.arr(1, Array);
            //int[][] tmp_count = new int[1][];
            var tmp_count = SVM.arr(1, Array);
            
            var perm = SVM.arr(l, 0);
            
            // group training data of the same class
            SVM.group_classes(prob,tmp_nr_class,tmp_label,tmp_start,tmp_count,perm);
            var nr_class = tmp_nr_class[0];            
            var label = tmp_label[0];
            var start = tmp_start[0];
            var count = tmp_count[0];
            //svm_node[][] x = new svm_node[l][];
            var x = SVM.arr(l, Array);
            
            var i;
            for(i=0; i<l; i++)
                x[i] = prob.x[perm[i]];

            // calculate weighted C

            var weighted_C = SVM.arr(nr_class, 0.0);
            
            for(i=0; i<nr_class; i++)
                weighted_C[i] = param.C;
            for(i=0; i<param.nr_weight; i++){
                var j;
                for(j=0;j<nr_class;j++)
                    if(param.weight_label[i] == label[j])
                        break;
                if(j == nr_class)
                    alert("warning: class label "+param.weight_label[i]+" specified in weight is not found\n");
                else
                    weighted_C[j] *= param.weight[i];
            }
            
            // train k*(k-1)/2 models
            var nonzero = SVM.arr(l, false);
            
            for(i=0; i<l; i++)
                nonzero[i] = false;
            f = SVM.arr(nr_class*(nr_class-1)/2, SVM.decision_function);
            
            var probA=null,probB = null;
            if (param.probability == 1){
                probA = SVM.arr(nr_class*(nr_class-1)/2, 0.0);
                probB = SVM.arr(nr_class*(nr_class-1)/2, 0.0);
            }
            
            var p = 0;
            for(i=0; i<nr_class; i++)
                for(var j=i+1; j<nr_class; j++)
                {
                    sub_prob = new SVM.Problem();
                    var si = start[i], sj = start[j];
                    var ci = count[i], cj = count[j];
                    sub_prob.l = ci+cj;
                    //sub_prob.x = new svm_node[sub_prob.l][];
                    
                    sub_prob.x = SVM.arr(sub_prob.l, Array);
                    
                    sub_prob.y = SVM.arr(sub_prob.l, 0.0);
                    
                    var k;
                    for(k=0;k<ci;k++){
                        sub_prob.x[k] = x[si+k];
                        sub_prob.y[k] = +1;
                    }
                    for(k=0; k<cj; k++){
                        sub_prob.x[ci+k] = x[sj+k];
                        sub_prob.y[ci+k] = -1;
                    }
                    
                    if(param.probability == 1){
                        var probAB = SVM.arr(2, 0.0);
                        svm_binary_svc_probability(sub_prob,param,weighted_C[i],weighted_C[j],probAB);
                        probA[p] = probAB[0];
                        probB[p] = probAB[1];
                    }
                    
                    f[p] = SVM.train_one(sub_prob, param, weighted_C[i], weighted_C[j]);
                    
                    for(k=0; k<ci; k++)
                        if(!nonzero[si+k] && Math.abs(f[p].alpha[k]) > 0)
                            nonzero[si+k] = true;
                    for(k=0;k<cj;k++)
                        if(!nonzero[sj+k] && Math.abs(f[p].alpha[ci+k]) > 0)
                            nonzero[sj+k] = true;
                    ++p;
                }
            
            // build output
            model.nr_class = nr_class;

            model.label = SVM.arr(nr_class, 0);
            for(i=0;i<nr_class;i++)
                model.label[i] = label[i];

            model.rho = SVM.arr(nr_class*(nr_class-1)/2, 0.0);
            for(i=0; i<nr_class * (nr_class-1) / 2; i++)
                model.rho[i] = f[i].rho;

            if(param.probability == 1){
                model.probA = SVM.arr(nr_class * (nr_class-1) / 2, 0.0);
                model.probB = SVM.arr(nr_class * (nr_class-1) / 2, 0.0);
                for(i=0; i<nr_class*(nr_class-1)/2; i++){
                    model.probA[i] = probA[i];
                    model.probB[i] = probB[i];
                }
            }
            else{
                model.probA = null;
                model.probB = null;
            }

            var nnz = 0;
            var nz_count = SVM.arr(nr_class, 0);
            model.nSV = SVM.arr(nr_class, 0);
            for(i=0; i<nr_class; i++){
                var nSV = 0;
                for(var j=0; j<count[i]; j++)
                    if(nonzero[start[i] + j]){
                        ++nSV;
                        ++nnz;
                    }
                model.nSV[i] = nSV;
                nz_count[i] = nSV;
            }

            SVM.info("Total nSV = "+nnz+"\n");

            model.l = nnz;
            //model.SV = new svm_node[nnz][];
            model.SV = SVM.arr(nnz, Array);

            p = 0;
            for(i=0;i<l;i++)
                if(nonzero[i]) model.SV[p++] = x[i];

            var nz_start = SVM.arr(nr_class, 0);
            nz_start[0] = 0;
            for(i=1; i<nr_class; i++)
                nz_start[i] = nz_start[i-1] + nz_count[i-1];

            //model.sv_coef = new double[nr_class-1][];
            model.sv_coef = SVM.arr(nr_class-1, Array);
            
            for(i=0;i<nr_class-1;i++)
                model.sv_coef[i] = SVM.arr(nnz, 0.0);

            p = 0;
            for(i=0; i<nr_class; i++)
                for(var j=i+1; j<nr_class; j++)
                {
                    // classifier (i,j): coefficients with
                    // i are in sv_coef[j-1][nz_start[i]...],
                    // j are in sv_coef[i][nz_start[j]...]

                    var si = start[i];
                    var sj = start[j];
                    var ci = count[i];
                    var cj = count[j];

                    var q = nz_start[i];
                    var k;
                    for(k=0; k<ci; k++)
                        if(nonzero[si+k])
                            model.sv_coef[j-1][q++] = f[p].alpha[k];
                    q = nz_start[j];
                    for(k=0; k<cj; k++)
                        if(nonzero[sj+k])
                            model.sv_coef[i][q++] = f[p].alpha[ci+k];
                    ++p;
                }
        }
        return model;
    }
    
    // Stratified cross validation
    SVM.cross_validation = function(prob, param, nr_fold, target)
    {
        var i;
        var fold_start = SVM.arr(nr_fold+1, 0);
        var l = prob.l;
        var perm = SVM.arr(l, 0);
        
        // stratified cv may not give leave-one-out rate
        // Each class to l folds -> some folds may have zero elements
        if((param.svm_type == SVM.C_SVC || param.svm_type == SVM.NU_SVC) && nr_fold < l)
        {
            var tmp_nr_class = SVM.arr(1, 0);
            //int[][] tmp_label = new int[1][];
            var tmp_label = SVM.arr(1, Array);
            //int[][] tmp_start = new int[1][];
            var tmp_start = SVM.arr(1, Array);
            //int[][] tmp_count = new int[1][];
            var tmp_count = SVM.arr(1, Array);
            

            SVM.group_classes(prob,tmp_nr_class,tmp_label,tmp_start,tmp_count,perm);

            var nr_class = tmp_nr_class[0];
            var start = tmp_start[0];
            var count = tmp_count[0];

            // random shuffle and then data grouped by fold using the array perm
            var fold_count = SVM.arr(nr_fold, 0);
            var c;
            var index = SVM.arr(l, 0);
            for(i=0; i<l; i++)
                index[i] = perm[i];
            for (c=0; c<nr_class; c++)
                for(i=0; i<count[c]; i++){
                    //var j = i+rand.nextInt(count[c]-i);
                    var j = i+SVM.nextInt(count[c]-i);
                    _=index[start[c]+j]; index[start[c]+j]=index[start[c]+i]; index[start[c]+i]=_;
                }
            for(i=0; i<nr_fold; i++){
                fold_count[i] = 0;
                for (c=0; c<nr_class; c++)
                    fold_count[i] += (i+1) * count[c] / nr_fold-i * count[c] / nr_fold;
            }
            fold_start[0]=0;
            for (i=1;i<=nr_fold;i++)
                fold_start[i] = fold_start[i-1]+fold_count[i-1];
            for (c=0; c<nr_class; c++)
                for(i=0; i<nr_fold; i++) {
                    var begin = start[c] + i * count[c] / nr_fold;
                    var end = start[c] + (i+1) * count[c] / nr_fold;
                    for(var j=begin; j<end; j++)
                    {
                        perm[fold_start[i]] = index[j];
                        fold_start[i]++;
                    }
                }
            fold_start[0] = 0;
            for (i=1; i<=nr_fold; i++)
                fold_start[i] = fold_start[i-1] + fold_count[i-1];
        }
        else {
            for(i=0;i<l;i++) perm[i]=i;
            for(i=0;i<l;i++){
                //var j = i + rand.nextInt(l-i);
                var j = i + SVM.nextInt(l-i);
                _=perm[i]; perm[i]=perm[j]; perm[j]=_;
            }
            for(i=0; i<=nr_fold; i++)
                fold_start[i] = i * l / nr_fold;
        }

        for(i=0; i<nr_fold; i++){
            var begin = fold_start[i];
            var end = fold_start[i+1];
            var j,k;
            subprob = new SVM.Problem();

            subprob.l = l-(end-begin);
            //subprob.x = new svm_node[subprob.l][];
            subprob.x = SVM.arr(subprob.l, Array); 
            
            subprob.x = SVM.arr(subprob.l, svm_node);
            subprob.y = SVM.arr(subprob.l, 0.0);

            k=0;
            for(j=0; j<begin; j++){
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for(j=end; j<l; j++){
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            submodel = SVM.train(subprob, param);
            if(param.probability==1 &&
               (param.svm_type == SVM.C_SVC ||
                param.svm_type == SVM.NU_SVC))
            {
                var prob_estimates= SVM.arr(SVM.get_nr_class(submodel), 0.0);
                for(j=begin; j<end; j++)
                    target[perm[j]] = SVM.predict_probability(submodel, prob.x[perm[j]], prob_estimates);
            }
            else
                for(j=begin; j<end; j++)
                    target[perm[j]] = SVM.predict(submodel, prob.x[perm[j]]);
        }
    }
    
    SVM.get_svm_type = function(model){ return model.param.svm_type; }
    SVM.get_nr_class = function(model){ return model.nr_class; }
    
    SVM.get_labels = function(model, label)
    {
        if (model.label != null)
            for(var i=0; i<model.nr_class; i++)
                label[i] = model.label[i];
    }
    
    SVM.get_svr_probability = function(model)
    {
        if ((model.param.svm_type == SVM.EPSILON_SVR || model.param.svm_type == SVM.NU_SVR) &&
            model.probA!=null)
        return model.probA[0];
        else
        {
            alert("Model doesn't contain information for SVR probability inference\n");
            return 0;
        }
    }
    
    SVM.predict_values = function(model, x, dec_values)
    {
        if(model.param.svm_type == SVM.ONE_CLASS ||
           model.param.svm_type == SVM.EPSILON_SVR ||
           model.param.svm_type == SVM.NU_SVR)
        {
            var sv_coef = model.sv_coef[0];
            var sum = 0;
            for(var i=0;i<model.l;i++)
                sum += sv_coef[i] * SVM.Kernel.k_function(x,model.SV[i],model.param);
            sum -= model.rho[0];
            dec_values[0] = sum;

            if(model.param.svm_type == SVM.ONE_CLASS)
                return (sum>0)?1:-1;
            else
                return sum;
        }
        else
        {
            var i;
            var nr_class = model.nr_class;
            var l = model.l;
        
            var kvalue = SVM.arr(l, 0.0);
            for(i=0;i<l;i++)
                kvalue[i] = SVM.Kernel.k_function(x,model.SV[i],model.param);

            var start = SVM.arr(nr_class, 0);
            start[0] = 0;
            for(i=1; i<nr_class; i++)
                start[i] = start[i-1]+model.nSV[i-1];

            var vote = SVM.arr(nr_class, 0);
            for(i=0; i<nr_class; i++)
                vote[i] = 0;

            var p=0;
            for(i=0;i<nr_class;i++)
                for(var j=i+1;j<nr_class;j++)
                {
                    var sum = 0;
                    var si = start[i];
                    var sj = start[j];
                    var ci = model.nSV[i];
                    var cj = model.nSV[j];
                
                    var k;
                    var coef1 = model.sv_coef[j-1];
                    var coef2 = model.sv_coef[i];
                    for(k=0; k<ci; k++)
                        sum += coef1[si+k] * kvalue[si+k];
                    for(k=0; k<cj; k++)
                        sum += coef2[sj+k] * kvalue[sj+k];
                    sum -= model.rho[p];
                    dec_values[p] = sum;                    

                    if(dec_values[p] > 0)
                        ++vote[i];
                    else
                        ++vote[j];
                    p++;
                }

            var vote_max_idx = 0;
            for(i=1; i<nr_class; i++)
                if(vote[i] > vote[vote_max_idx])
                    vote_max_idx = i;

            return model.label[vote_max_idx];
        }
    }
    
    SVM.predict = function(model, x)
    {
        var nr_class = model.nr_class;
        var dec_values = null;
        if(model.param.svm_type == SVM.ONE_CLASS ||
                model.param.svm_type == SVM.EPSILON_SVR ||
                model.param.svm_type == SVM.NU_SVR)
            dec_values = SVM.arr(1, 0.0);
        else
            dec_values = SVM.arr(nr_class*(nr_class-1)/2, 0.0);
        var pred_result = SVM.predict_values(model, x, dec_values);
        return pred_result;
    }
    
    SVM.predict_probability = function(model, x, prob_estimates)
    {
        if ((model.param.svm_type == SVM.C_SVC || model.param.svm_type == SVM.NU_SVC) &&
            model.probA!=null && model.probB!=null)
        {
            var i;
            var nr_class = model.nr_class;
            var dec_values = SVM.arr(nr_class*(nr_class-1)/2, 0.0);
            SVM.predict_values(model, x, dec_values);

            var min_prob=1e-7;
            var pairwise_prob = SVM.arr([nr_class,nr_class], 0.0);
            
            var k=0;
            for(i=0; i<nr_class; i++)
                for(var j=i+1; j<nr_class; j++)
                {
                    pairwise_prob[i][j] = Math.min(Math.max(sigmoid_predict(dec_values[k],model.probA[k],model.probB[k]),min_prob),1-min_prob);
                    pairwise_prob[j][i] = 1-pairwise_prob[i][j];
                    k++;
                }
            SVM.multiclass_probability(nr_class,pairwise_prob,prob_estimates);

            var prob_max_idx = 0;
            for(i=1;i<nr_class;i++)
                if(prob_estimates[i] > prob_estimates[prob_max_idx])
                    prob_max_idx = i;
            return model.label[prob_max_idx];
        }
        else 
            return SVM.predict(model, x);
    }
    
    //TODO: svm_save_model
    //TODO: svm_load_model
    
    SVM.check_parameter = function(prob, param)
    {
        //svm_type
        var svm_type = param.svm_type;
        if(svm_type != SVM.C_SVC &&
        svm_type != SVM.NU_SVC &&
           svm_type != SVM.ONE_CLASS &&
           svm_type != SVM.EPSILON_SVR &&
           svm_type != SVM.NU_SVR)
        return "unknown svm type";
        
        //kernel type
        var kernel_type = param.kernel_type;
        if(kernel_type != SVM.LINEAR &&
           kernel_type != SVM.POLY &&
           kernel_type != SVM.RBF &&
           kernel_type != SVM.SIGMOID &&
           kernel_type != SVM.PRECOMPUTED)
        return "unknown kernel type";
        
        if(param.gamma < 0)
            return "gamma < 0";

        if(param.degree < 0)
            return "degree of polynomial kernel < 0";
            
        // cache_size,eps,C,nu,p,shrinking
        if(param.cache_size <= 0)
            return "cache_size <= 0";

        if(param.eps <= 0)
            return "eps <= 0";

        if(svm_type == SVM.C_SVC ||
           svm_type == SVM.EPSILON_SVR ||
           svm_type == SVM.NU_SVR)
            if(param.C <= 0)
                return "C <= 0";
        
        if(svm_type == SVM.NU_SVC ||
           svm_type == SVM.ONE_CLASS ||
           svm_type == SVM.NU_SVR)
            if(param.nu <= 0 || param.nu > 1)
                return "nu <= 0 or nu > 1";

        if(svm_type == SVM.EPSILON_SVR)
            if(param.p < 0)
                return "p < 0";

        if(param.shrinking != 0 &&
           param.shrinking != 1)
            return "shrinking != 0 and shrinking != 1";

        if(param.probability != 0 &&
           param.probability != 1)
            return "probability != 0 and probability != 1";
        
        if(param.probability == 1 &&
           svm_type == SVM.ONE_CLASS)
            return "one-class SVM probability output not supported yet";
        
        // check whether nu-svc is feasible
        if(svm_type == SVM.NU_SVC)
        {
            var l = prob.l;
            var max_nr_class = 16;
            var nr_class = 0;
            var label = SVM.arr(max_nr_class, 0);
            var count = SVM.arr(max_nr_class, 0);

            
            for(var i=0; i<l; i++){
                var this_label = prob.y[i];
                var j;
                for(j=0; j<nr_class; j++)
                    if(this_label == label[j]){
                        ++count[j];
                        break;
                    }

                if(j == nr_class){
                    if(nr_class == max_nr_class){
                        max_nr_class *= 2;
                        var new_data = SVM.arr(max_nr_class, 0);
                        //System.arraycopy(label,0,new_data,0,label.length);
                        new_data = label.slice(0);
                        label = new_data;
                        
                        new_data = SVM.arr(max_nr_class, 0);
                        //System.arraycopy(count,0,new_data,0,count.length);
                        new_data = count.slice(0);
                        count = new_data;
                    }
                    label[nr_class] = this_label;
                    count[nr_class] = 1;
                    ++nr_class;
                }
            }

            for(i=0; i<nr_class; i++){
                var n1 = count[i];
                for(var j=i+1;j<nr_class;j++){
                    var n2 = count[j];
                    if(param.nu*(n1+n2)/2 > Math.min(n1,n2))
                        return "specified nu is infeasible";
                }
            }
        }
        
        return null;
        
    }
    
}());

}());