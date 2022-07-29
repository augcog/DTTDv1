/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A base protocol for Metal views.
*/

import SwiftUI
import MetalKit
import Metal

/**
The `MetalRepresentable` protocol extends `UIViewRepresentable` to allow `MTKView` objects in SwiftUI.
The `rotationAngle` presents the camera streams upright on device rotations.
Each new metal view that conforms to `MetalRepresentable` is required to add its needed input properties,
and to implement the `makeCoordinator` function to return the coordinator that holds the view's drawing logic.
*/
protocol MetalRepresentable: UIViewRepresentable {
    var rotationAngle: Double { get set }
}

/// An extension of `MetalRepresentable` to share the settings for the conforming views.
extension MetalRepresentable where Self.Coordinator: MTKCoordinator<Self> {
    
    func makeUIView(context: UIViewRepresentableContext<Self>) -> MTKView {
        let mtkView = MTKView()
        mtkView.delegate = context.coordinator
        mtkView.backgroundColor = context.environment.colorScheme == .dark ? .black : .white
        context.coordinator.setupView(mtkView: mtkView)
        return mtkView
    }
    
    // Fix the rotation of the view by rotating the view by the rotation angle.
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<Self>) {
        view.transform = CGAffineTransform(rotationAngle: rotationAngle)
    }
    
}

/**
 The base coordinator class that conforms to `MTKViewDelegate`. Subclasses should override:
 - `preparePipelineAndDepthState()` - to create a pipeline descriptor with the required vertex and fragment
                                      function to create a `pipelineState` and `depthState` if needed.
- `draw()` - to perform the drawing operation.
 */
class MTKCoordinator<MTKViewRepresentable: MetalRepresentable>: NSObject, MTKViewDelegate {
    
    weak var mtkView: MTKView!
    
    var pipelineState: MTLRenderPipelineState!
    var metalCommandQueue: MTLCommandQueue
    var depthState: MTLDepthStencilState!
    var parent: MTKViewRepresentable
    
    init(parent: MTKViewRepresentable) {
        self.parent = parent
        self.metalCommandQueue = MetalEnvironment.shared.metalCommandQueue
        super.init()
    }
    
    /// Saves a reference to the `MTKView` in the coordinator and sets up the default settings.
    func setupView(mtkView: MTKView) {
        self.mtkView = mtkView
        self.mtkView.preferredFramesPerSecond = 60
        self.mtkView.isOpaque = true
        self.mtkView.framebufferOnly = false
        self.mtkView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        self.mtkView.drawableSize = mtkView.frame.size
        self.mtkView.enableSetNeedsDisplay = false
        self.mtkView.colorPixelFormat = .bgra8Unorm
        self.mtkView.depthStencilPixelFormat = .depth32Float
        self.mtkView.contentMode = .scaleAspectFit
        self.mtkView.device = MetalEnvironment.shared.metalDevice
        preparePipelineAndDepthState()
    }
    
    /// The app uses a quad to draw a texture onscreen. It creates a `MTLVertexDescriptor` for this case.
    func createPlaneMetalVertexDescriptor() -> MTLVertexDescriptor {
        let mtlVertexDescriptor: MTLVertexDescriptor = MTLVertexDescriptor()
        // Store position in `attribute[[0]]`.
        mtlVertexDescriptor.attributes[0].format = .float2
        mtlVertexDescriptor.attributes[0].offset = 0
        mtlVertexDescriptor.attributes[0].bufferIndex = 0
        
        // Store texture coordinates in `attribute[[1]]`.
        mtlVertexDescriptor.attributes[1].format = .float2
        mtlVertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
        mtlVertexDescriptor.attributes[1].bufferIndex = 0
        
        // Set stride to twice the `float2` bytes per vertex.
        mtlVertexDescriptor.layouts[0].stride = 2 * MemoryLayout<SIMD2<Float>>.stride
        mtlVertexDescriptor.layouts[0].stepRate = 1
        mtlVertexDescriptor.layouts[0].stepFunction = .perVertex
        
        return mtlVertexDescriptor
    }
    
    func preparePipelineAndDepthState() {}
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        // Override in subclass.
    }
    
    func draw(in view: MTKView) {
        // Override in subclass.
    }
}

