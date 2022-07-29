/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A view that draws depth textures.
*/

import Foundation
import SwiftUI
import MetalKit
import Metal

struct MetalTextureDepthView: UIViewRepresentable, MetalRepresentable {
    var rotationAngle: Double

    @Binding var maxDepth: Float
    @Binding var minDepth: Float
    var capturedData: CameraCapturedData
    
    func makeCoordinator() -> MTKDepthTextureCoordinator {
        MTKDepthTextureCoordinator(parent: self)
    }
}

final class MTKDepthTextureCoordinator: MTKCoordinator<MetalTextureDepthView> {
    override func preparePipelineAndDepthState() {
        guard let metalDevice = mtkView.device else { fatalError("Expected a Metal device.") }
        do {
            let library = MetalEnvironment.shared.metalLibrary
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            pipelineDescriptor.vertexFunction = library.makeFunction(name: "planeVertexShader")
            pipelineDescriptor.fragmentFunction = library.makeFunction(name: "planeFragmentShaderDepth")
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
        guard parent.capturedData.depth != nil else {
            print("There's no content to display.")
            return
        }
        guard let commandBuffer = metalCommandQueue.makeCommandBuffer() else { return }
        guard let passDescriptor = view.currentRenderPassDescriptor else { return }
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDescriptor) else { return }
        // Vertex and Texture coordinates data (x,y,u,v) * 4 ordered for triangle strip
        let vertexData: [Float] = [  -1, -1, 1, 1,
                                     1, -1, 1, 0,
                                     -1, 1, 0, 1,
                                     1, 1, 0, 0]
        encoder.setVertexBytes(vertexData, length: vertexData.count * MemoryLayout<Float>.stride, index: 0)
        encoder.setFragmentBytes(&parent.minDepth, length: MemoryLayout<Float>.stride, index: 0)
        encoder.setFragmentBytes(&parent.maxDepth, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setFragmentTexture(parent.capturedData.depth!, index: 0)
        encoder.setDepthStencilState(depthState)
        encoder.setRenderPipelineState(pipelineState)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.present(view.currentDrawable!)
        commandBuffer.commit()
    }

}

