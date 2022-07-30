/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view that draws a jet colored depth overlay on the color image
 between maxDepth and minDepth with a wave effect that moves along the Z axis.
*/

import SwiftUI
import Combine
import MetalKit
import Metal

struct MetalTextureColorZapView: UIViewRepresentable, MetalRepresentable {
    var rotationAngle: Double
    @Binding var maxDepth: Float
    @Binding var minDepth: Float

    var capturedData: CameraCapturedData
    
    func makeCoordinator() -> MTKColorZapCoordinator {
        MTKColorZapCoordinator(parent: self)
    }
}

final class MTKColorZapCoordinator: MTKCoordinator<MetalTextureColorZapView> {
    var iTime: Int = 0
    let speedFactor: Float = 0.01
    override func preparePipelineAndDepthState() {
        guard let metalDevice = mtkView.device else { fatalError("Expected a Metal device.") }
        do {
            let library = MetalEnvironment.shared.metalLibrary
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.vertexFunction = library.makeFunction(name: "planeVertexShader")
            pipelineDescriptor.fragmentFunction = library.makeFunction(name: "planeFragmentShaderColorZap")
            pipelineDescriptor.vertexDescriptor = createPlaneMetalVertexDescriptor()
            pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
            pipelineState = try metalDevice.makeRenderPipelineState(descriptor: pipelineDescriptor)
            
            let depthDescriptor = MTLDepthStencilDescriptor()
            depthDescriptor.isDepthWriteEnabled = true
            depthDescriptor.depthCompareFunction = .less
            depthState = metalDevice.makeDepthStencilState(descriptor: depthDescriptor)
        } catch {
            print("Unexpected error: \(error).")
        }
    }
    
    override func draw(in view: MTKView) {
        guard parent.capturedData.colorY != nil && parent.capturedData.colorCbCr != nil else {
            print("There's no content to display.")
            return
        }
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer() else { return }
        guard let passDescriptor = view.currentRenderPassDescriptor else { return }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else { return }
        // Vertex and Texture coordinates data (x,y,u,v) * 4 ordered for triangle strip
        let vertexData: [Float] = [-1, -1, 1, 1,
                                    1, -1, 1, 0,
                                   -1,  1, 0, 1,
                                    1,  1, 0, 0]
        let f = Float(abs(sin(Float(iTime) * speedFactor)))
        var minDepth: Float = self.parent.minDepth
        var maxDepth: Float = (self.parent.maxDepth - minDepth ) * f + minDepth
        encoder.setVertexBytes(vertexData, length: vertexData.count * MemoryLayout<Float>.stride, index: 0)
        encoder.setFragmentBytes(&minDepth, length: MemoryLayout<Float>.stride, index: 0)
        encoder.setFragmentBytes(&maxDepth, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setFragmentBytes(&self.parent.maxDepth, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setFragmentTexture(parent.capturedData.depth!, index: 2)
        encoder.setFragmentTexture(parent.capturedData.colorY!, index: 0)
        encoder.setFragmentTexture(parent.capturedData.colorCbCr!, index: 1)
        encoder.setDepthStencilState(depthState)
        encoder.setRenderPipelineState(pipelineState)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
        iTime += 1
    }
}
