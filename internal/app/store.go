package app

import (
	"strconv"
	"sync"
)

type storedResponse struct {
	EffectiveInput, Context []any
	Response                map[string]any
}
type responseStore struct {
	mu     sync.RWMutex
	max    int
	order  []string
	values map[string]*storedResponse
}

func newResponseStore(max int) *responseStore {
	return &responseStore{max: max, values: map[string]*storedResponse{}}
}
func (s *responseStore) get(id string) *storedResponse {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v := s.values[id]
	if v == nil {
		return nil
	}
	return &storedResponse{EffectiveInput: cloneSlice(v.EffectiveInput), Context: cloneSlice(v.Context), Response: cloneMap(v.Response)}
}
func (s *responseStore) remember(id string, input []any, response map[string]any) {
	if s.max == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.values[id]; !ok {
		s.order = append(s.order, id)
	}
	effective := cloneSlice(input)
	s.values[id] = &storedResponse{EffectiveInput: effective, Context: append(cloneSlice(effective), responseContext(response)...), Response: cloneMap(response)}
	for len(s.order) > s.max {
		delete(s.values, s.order[0])
		s.order = s.order[1:]
	}
}
func (s *responseStore) delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.values[id]; !ok {
		return false
	}
	delete(s.values, id)
	for i, v := range s.order {
		if v == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	return true
}
func (s *responseStore) cancel(id string) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v := s.values[id]; v != nil {
		v.Response["status"] = "cancelled"
		return cloneMap(v.Response)
	}
	return nil
}

type storedChat struct {
	Completion map[string]any
	Messages   []map[string]any
	Metadata   map[string]any
}
type chatStore struct {
	mu     sync.RWMutex
	max    int
	order  []string
	values map[string]*storedChat
}

func newChatStore(max int) *chatStore { return &chatStore{max: max, values: map[string]*storedChat{}} }
func (s *chatStore) get(id string) *storedChat {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v := s.values[id]
	if v == nil {
		return nil
	}
	result := &storedChat{Completion: cloneMap(v.Completion), Metadata: cloneMap(v.Metadata)}
	for _, m := range v.Messages {
		result.Messages = append(result.Messages, cloneMap(m))
	}
	return result
}
func (s *chatStore) remember(id string, completion, metadata map[string]any) {
	if s.max == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	c := cloneMap(completion)
	c["metadata"] = cloneMap(metadata)
	var messages []map[string]any
	for i, raw := range sliceAny(c["choices"]) {
		choice := mapAny(raw)
		m := cloneMap(mapAny(choice["message"]))
		if m != nil {
			m["id"] = id + "_msg_" + strconv.Itoa(i)
			messages = append(messages, m)
		}
	}
	if _, ok := s.values[id]; !ok {
		s.order = append(s.order, id)
	}
	s.values[id] = &storedChat{Completion: c, Messages: messages, Metadata: cloneMap(metadata)}
	for len(s.order) > s.max {
		delete(s.values, s.order[0])
		s.order = s.order[1:]
	}
}
func (s *chatStore) update(id string, metadata map[string]any) map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	v := s.values[id]
	if v == nil {
		return nil
	}
	v.Metadata = cloneMap(metadata)
	v.Completion["metadata"] = cloneMap(metadata)
	return cloneMap(v.Completion)
}
func (s *chatStore) delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.values[id]; !ok {
		return false
	}
	delete(s.values, id)
	for i, v := range s.order {
		if v == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
	}
	return true
}
func (s *chatStore) list(model string, metadata map[string]any, desc bool, after string, limit int) ([]map[string]any, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := append([]string(nil), s.order...)
	if desc {
		for i, j := 0, len(ids)-1; i < j; i, j = i+1, j-1 {
			ids[i], ids[j] = ids[j], ids[i]
		}
	}
	if after != "" {
		for i, id := range ids {
			if id == after {
				ids = ids[i+1:]
				break
			}
		}
	}
	var result []map[string]any
	for _, id := range ids {
		v := s.values[id]
		if model != "" && stringValue(v.Completion["model"]) != model {
			continue
		}
		match := true
		for k, want := range metadata {
			if stringValue(v.Metadata[k]) != stringValue(want) {
				match = false
			}
		}
		if match {
			result = append(result, cloneMap(v.Completion))
		}
	}
	more := limit > 0 && len(result) > limit
	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}
	return result, more
}
