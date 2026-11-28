//! What the translated shaders need of the host's subgroups (`docs/shaders.md`, As built (7c)):
//! the one list of it, which the device choice checks, and what happens to a module on a device
//! that offers less than the whole.
//!
//! * Required of every device: [`OPERATIONS`] in the [`STAGES`], and [`MIN_SIZE`] invocations
//!   per subgroup (quads).
//! * Per pipeline: a stage outside [`STAGES`] (vertex) may use subgroup operations only where
//!   the device supports them there (lavapipe does not).
//! * Optional: `VK_KHR_shader_maximal_reconvergence`. It is enabled when present; without it
//!   the declaration is removed from the modules, whose subgroup operations then act on
//!   whatever invocations the driver's own reconvergence leaves together in divergent flow.
//!
//! No feature bit, and `VK_EXT_subgroup_size_control` is not used: nothing asks for a size or
//! for full subgroups, so an operation covers at most `subgroupSize` invocations, and a device
//! with several sizes may run a stage narrower (Intel: 16 of 32 seen). A console wave of 64 is
//! therefore not one subgroup on such hosts (`docs/shaders.md`, As built (7c)).

use super::log_once;
use ash::{khr, vk, Instance};
use spirv::{Capability, ExecutionMode, Op};

/// The `VkSubgroupFeatureFlagBits` the emitter's capabilities stand for.
const OPERATIONS: [(vk::SubgroupFeatureFlags, &str); 6] = [
    (vk::SubgroupFeatureFlags::BASIC, "basic"),
    (vk::SubgroupFeatureFlags::VOTE, "vote"),
    (vk::SubgroupFeatureFlags::ARITHMETIC, "arithmetic"),
    (vk::SubgroupFeatureFlags::BALLOT, "ballot"),
    (vk::SubgroupFeatureFlags::SHUFFLE, "shuffle"),
    (vk::SubgroupFeatureFlags::QUAD, "quad"),
];

/// The stages every device must run them in.
const STAGES: [(vk::ShaderStageFlags, &str); 2] = [
    (vk::ShaderStageFlags::COMPUTE, "compute"),
    (vk::ShaderStageFlags::FRAGMENT, "fragment"),
];

/// A quad is four invocations of one subgroup.
const MIN_SIZE: u32 = 4;

const MAX_RECONVERGENCE: &[u8] = b"SPV_KHR_maximal_reconvergence";

/// The subgroup properties of a physical device.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Subgroups {
    pub size: u32,
    stages: vk::ShaderStageFlags,
    operations: vk::SubgroupFeatureFlags,
    quad_in_all_stages: bool,
    /// `VK_KHR_shader_maximal_reconvergence` and its feature are there.
    pub max_reconvergence: bool,
}

impl Subgroups {
    /// `phys` must be a Vulkan 1.2 device or later.
    pub(crate) fn query(instance: &Instance, phys: vk::PhysicalDevice) -> Subgroups {
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p11);
        let mut mr = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
        // SAFETY: queries on a physical device of this instance; the feature struct is only
        // chained when the device has its extension.
        unsafe {
            instance.get_physical_device_properties2(phys, &mut p2);
            let ext = instance
                .enumerate_device_extension_properties(phys)
                .unwrap_or_default()
                .iter()
                .any(|e| {
                    e.extension_name_as_c_str() == Ok(khr::shader_maximal_reconvergence::NAME)
                });
            if ext {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut mr);
                instance.get_physical_device_features2(phys, &mut f2);
            }
        }
        Subgroups {
            size: p11.subgroup_size,
            stages: p11.subgroup_supported_stages,
            operations: p11.subgroup_supported_operations,
            quad_in_all_stages: p11.subgroup_quad_operations_in_all_stages != 0,
            max_reconvergence: mr.shader_maximal_reconvergence != 0,
        }
    }

    /// The first requirement the device does not meet.
    pub(crate) fn missing(&self) -> Option<String> {
        if let Some((_, n)) = OPERATIONS
            .iter()
            .find(|(f, _)| !self.operations.contains(*f))
        {
            return Some(format!("subgroup {n} operations"));
        }
        if let Some((_, n)) = STAGES.iter().find(|(f, _)| !self.stages.contains(*f)) {
            return Some(format!("subgroup operations in {n} shaders"));
        }
        (self.size < MIN_SIZE).then(|| {
            format!(
                "subgroups of {MIN_SIZE} invocations or more (subgroupSize {})",
                self.size
            )
        })
    }

    /// Makes a translated module of `stage` fit the device, or says why it cannot run there.
    pub(crate) fn adapt(
        &self,
        words: &mut Vec<u32>,
        stage: vk::ShaderStageFlags,
    ) -> Result<(), String> {
        let (mut uses, mut quad) = (false, false);
        for i in instructions(words) {
            if i[0] & 0xffff == Op::Capability as u32 {
                let c = i[1];
                uses |= (Capability::GroupNonUniform as u32
                    ..=Capability::GroupNonUniformQuad as u32)
                    .contains(&c);
                quad |= c == Capability::GroupNonUniformQuad as u32;
            }
        }
        if !uses {
            return Ok(());
        }
        let always = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;
        if !self.stages.contains(stage) {
            return Err(format!(
                "subgroup operations in a {stage:?} shader, which this device lacks"
            ));
        }
        if quad && !always.contains(stage) && !self.quad_in_all_stages {
            return Err(format!(
                "quad operations in a {stage:?} shader, which this device lacks \
                 (quadOperationsInAllStages)"
            ));
        }
        if !self.max_reconvergence {
            log_once(
                "no VK_KHR_shader_maximal_reconvergence: subgroup operations in divergent \
                 flow rely on the driver's reconvergence"
                    .into(),
            );
            strip_max_reconvergence(words);
        }
        Ok(())
    }
}

/// The instructions of a module, after its five header words.
fn instructions(words: &[u32]) -> impl Iterator<Item = &[u32]> {
    let mut rest = words.get(5..).unwrap_or_default();
    std::iter::from_fn(move || {
        let len = (*rest.first()? >> 16) as usize;
        if len == 0 || len > rest.len() {
            return None;
        }
        let (i, tail) = rest.split_at(len);
        rest = tail;
        Some(i)
    })
}

/// Removes the `OpExtension` and the execution mode of maximal reconvergence.
fn strip_max_reconvergence(words: &mut Vec<u32>) {
    let is_name = |w: &[u32]| {
        let bytes = w.iter().flat_map(|w| w.to_le_bytes());
        bytes
            .take_while(|&b| b != 0)
            .eq(MAX_RECONVERGENCE.iter().copied())
    };
    let mut out = words[..5.min(words.len())].to_vec();
    for i in instructions(words) {
        let op = i[0] & 0xffff;
        let declares = (op == Op::Extension as u32 && is_name(&i[1..]))
            || (op == Op::ExecutionMode as u32
                && i.get(2) == Some(&(ExecutionMode::MaximallyReconvergesKHR as u32)));
        if !declares {
            out.extend_from_slice(i);
        }
    }
    *words = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Subgroups {
        Subgroups {
            size: 32,
            stages: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT,
            operations: OPERATIONS.iter().fold(Default::default(), |a, o| a | o.0),
            quad_in_all_stages: false,
            max_reconvergence: true,
        }
    }

    fn inst(op: Op, operands: &[u32]) -> Vec<u32> {
        let mut v = vec![((operands.len() as u32 + 1) << 16) | op as u32];
        v.extend_from_slice(operands);
        v
    }

    /// A header, the quad capability, the extension and the execution mode.
    fn module() -> Vec<u32> {
        let mut name = MAX_RECONVERGENCE.to_vec();
        name.resize((name.len() / 4 + 1) * 4, 0);
        let name: Vec<u32> = name
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut m = vec![spirv::MAGIC_NUMBER, 0x0001_0500, 0, 9, 0];
        m.extend(inst(
            Op::Capability,
            &[Capability::GroupNonUniformQuad as u32],
        ));
        m.extend(inst(Op::Extension, &name));
        m.extend(inst(
            Op::ExecutionMode,
            &[1, ExecutionMode::MaximallyReconvergesKHR as u32],
        ));
        m
    }

    #[test]
    fn the_first_missing_requirement_is_named() {
        assert_eq!(all().missing(), None);
        let mut s = all();
        s.operations = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::VOTE;
        assert_eq!(s.missing().unwrap(), "subgroup arithmetic operations");
        let mut s = all();
        s.stages = vk::ShaderStageFlags::COMPUTE;
        assert_eq!(
            s.missing().unwrap(),
            "subgroup operations in fragment shaders"
        );
        let mut s = all();
        s.size = 1;
        assert!(s.missing().unwrap().contains("subgroupSize 1"));
    }

    #[test]
    fn a_module_is_refused_or_stripped_to_fit_the_device() {
        let mut m = module();
        all().adapt(&mut m, vk::ShaderStageFlags::COMPUTE).unwrap();
        assert_eq!(m, module());
        let e = all()
            .adapt(&mut m, vk::ShaderStageFlags::VERTEX)
            .unwrap_err();
        assert!(e.contains("VERTEX"), "{e}");
        let mut s = all();
        s.stages |= vk::ShaderStageFlags::VERTEX;
        let e = s.adapt(&mut m, vk::ShaderStageFlags::VERTEX).unwrap_err();
        assert!(e.contains("quadOperationsInAllStages"), "{e}");
        let mut s = all();
        s.max_reconvergence = false;
        s.adapt(&mut m, vk::ShaderStageFlags::COMPUTE).unwrap();
        assert_eq!(m, module()[..7]);
        // a module without subgroup operations fits any stage
        let mut plain = module()[..5].to_vec();
        Subgroups::default()
            .adapt(&mut plain, vk::ShaderStageFlags::VERTEX)
            .unwrap();
    }
}
