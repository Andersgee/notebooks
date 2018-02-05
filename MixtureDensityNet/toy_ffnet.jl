module nn
#for toying and experimentation
#assumes length(Lw) is 3
function fprop!(x, W,b,z,y)
    A_mul_B!(z[1], W[1], x); z[1].+=b[1]
    @. y[1] = tanh(z[1])
    A_mul_B!(z[2], W[2], y[1]); z[2].+=b[2]
    @. y[2] = 1*z[2]
end

function inference(x,W,b)
    return W[2]*(tanh.(W[1]*x.+b[1])).+b[2]
end

function ∇MSE!(∇z, y,t)
    @. ∇z[2] = y[2]-t
end

function bprop!(∇W,∇z,∇y, W,x,y)
    A_mul_Bt!(∇W[2], ∇z[2], y[1])
    At_mul_B!(∇y[1], W[2], ∇z[2])    
    @. ∇z[1] = ∇y[1] * (1-y[1]^2)
    A_mul_Bt!(∇W[1], ∇z[1], x)
end

function adjust!(W,b, ∇W,∇z, α)
    for n=1:length(b)
        @. W[n] -= α*∇W[n]
        b[n] .-= α*sum(∇z[n],2)
    end
end

function preallocate(Lw, bsz)
    N = length(Lw)-1
        
    W = [randn(Lw[n+1],Lw[n])./sqrt(Lw[n]) for n=1:N]
    ∇W = deepcopy(W)
    b = [zeros(Lw[n+1],1) for n=1:N]
    ∇b = deepcopy(b)
    
    z = [zeros(Lw[n+1],bsz) for n=1:N]
    ∇z = deepcopy(z)
    y = [zeros(Lw[n+1],bsz) for n=1:N]
    ∇y = deepcopy(y)
    return W,b,z,y, ∇W,∇b,∇z,∇y
end

function softmax(x)
    m=maximum(x,1)
    p = exp.(x .- m)
    return p ./ sum(p,1)
end

function sigm(x)
    1./(1+exp.(-x))
end

end